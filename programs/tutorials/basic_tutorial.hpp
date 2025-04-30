#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#define RESULTTYPE \
    DATATYPE  // Each main function should define this to make a result accessible after the MPC protocol has finished
#define FUNCTION Basic_Tutorial  // define your main entry point

template <typename Share>  // Share is defined by the MPC protocol specified in config.h
void Basic_Tutorial(
    DATATYPE* res)  // The main entry pint needs to define a result pointer, DATATYPE is specified in config.h
{

    //---------------------------------//
    // Basic Datatypes
    using A = Additive_Share<DATATYPE, Share>;  // Additive Share to store secret shares in the arithmetic domain
    using S = XOR_Share<DATATYPE, Share>;       // XOR Share to store secret shares in the boolean domain
    UINT_TYPE integer_value =
        5;  // Plaintext integer with the same BITLENGTH as secret shares, BITLENGTH is defined in config.h
    UINT_TYPE fixed_point_value = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(
        3.5f);  // Plaintext fixed point value with FRACTIONAL bits of precision, defined in config.h

    //---------------------------------//
    // The vectorized programming model
    const int vectorization_factor =
        DATTYPE / BITLENGTH;  // Each instruction in the framework is inherently vectorized.
    alignas(sizeof(DATATYPE)) UINT_TYPE input_of_party_0[vectorization_factor];
#if PSELF == P_0  // only execute this code if the party is P0
    for (int i = 0; i < vectorization_factor; i++)
        input_of_party_0[i] = i;  // could also read from a file or any other source
#endif
    DATATYPE vectorized_input_of_party_0;
    orthogonalize_arithmetic(
        input_of_party_0,
        &vectorized_input_of_party_0,
        1);  // A set of inputs gets vectorized in a single variable depending on the chosen registerlength

    //---------------------------------//
    // Secret Sharing

    // Private Inputs with vectorization
    A secret;
    secret.template prepare_receive_from<P_0>(vectorized_input_of_party_0);  // Party 0 prepares the secret share
    Share::communicate();                          // Communication needs to be triggered explicitly
    secret.template complete_receive_from<P_0>();  // The secret is now shared among all parties

    // Public Inputs with vectorization
    A public_share;  //
    UINT_TYPE vectorized_public_input[vectorization_factor];
    DATATYPE public_input;
    for (int i = 0; i < vectorization_factor; i++)
        vectorized_public_input[i] = i;
    orthogonalize_arithmetic(
        vectorized_public_input,
        &public_input,
        1);  // A set of inputs gets vectorized in a single variable depending on the chosen registerlength
    public_share = A::get_share_from_public_dat(public_input);  // A public value is shared among all parties

    // Identical private inputs
    A single_secret;
    single_secret.template prepare_receive_from<P_1>(
        PROMOTE(9));  // Party 1 prepares the secret share of the value 9, PROMOTE ensures all vectorized values are set
                      // to the same single value
    Share::communicate();
    single_secret.template complete_receive_from<P_1>();  // The secret is now shared among all parties

    // Identical public inputs
    A another_public_share =
        2;  // A public value is shared among all parties, all vectorized values are set to the same single value

    //---------------------------------//
    // Arithmetic Operations on secret shares
    A result = secret + public_share;      // Addition
    result = result - public_share;        // Subtraction
    result = result.prepare_mult(secret);  // Prepare multiplication
    Share::communicate();                  // Communication needs to be triggered explicitly
    result.complete_mult_without_trunc();  // Complete multiplication, truncation is not required for integer
                                           // multiplication

    //---------------------------------//
    // Arithmetic Operations with public values
    result = result + A(8);          // Addition
    result = result - A(4);          // Subtraction
    result = result.mult_public(2);  // Multiplication with a public value

    //---------------------------------//
    // Revealing the result

    DATATYPE output;
    result.prepare_reveal_to_all();
    Share::communicate();
    output = result.complete_reveal_to_all();              // Reveals the result to all parties
    UINT_TYPE output_array[vectorization_factor];          // the result is still vectorized
    unorthogonalize_arithmetic(&output, output_array, 1);  // unvectorize the result
    for (int i = 0; i < vectorization_factor; i++)
        print_online("Output " + std::to_string(i) + ": " + std::to_string(output_array[i]) +
                     "\n");  // Print the result in the online phase

    //---------------------------------//
    // Minimizing communication rounds

    // Always process as many independent operations as possible before communicating

    const int len = 10;
    A array_a[len];
    A array_b[len];
    A arrary_c[len];
    for (int i = 0; i < len; i++)
    {
        array_a[i].template prepare_receive_from<P_0>(PROMOTE(i));        // Prepare secret shares of the values 0 to 9
        array_b[i].template prepare_receive_from<P_1>(PROMOTE(len - i));  // Prepare secret shares of the values 10 to 1
    }
    Share::communicate();  // Communicate once for all inputs
    for (int i = 0; i < len; i++)
    {
        array_a[i].template complete_receive_from<P_0>();  // Make sure the order of the completions matches the order
                                                           // of the prepare calls
        array_b[i].template complete_receive_from<P_1>();
    }
    for (int i = 0; i < len; i++)
        arrary_c[i] =
            array_a[i].prepare_mult(array_b[i]);  // Only after complete operations are done, shares should be accessed
    Share::communicate();
    for (int i = 0; i < len; i++)
        arrary_c[i]
            .complete_mult_without_trunc();  // Only after complete operations are done, shares should be accessed

    for (int i = 0; i < len; i++)
        arrary_c[i].prepare_reveal_to_all();
    Share::communicate();
    DATATYPE result_array[len];
    for (int i = 0; i < len; i++)
        result_array[i] = arrary_c[i].complete_reveal_to_all();  // Reveals the result to all parties
    alignas(sizeof(DATATYPE))
        UINT_TYPE result_array_vectorized[len][vectorization_factor];               // the result is still vectorized
    unorthogonalize_arithmetic(result_array, &result_array_vectorized[0][0], len);  // unvectorize the result
    for (int i = 0; i < len; i++)
        for (int j = 0; j < vectorization_factor; j++)
            print_online("Output " + std::to_string(i) + "," + std::to_string(j) + ": " +
                         std::to_string(result_array_vectorized[i][j]) + "\n");  // Print the result in the online phase
}
