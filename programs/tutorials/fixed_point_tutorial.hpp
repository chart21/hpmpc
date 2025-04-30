#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/prob_div.hpp"  // The functions directory contains multiple high-level MPC primitives
#define RESULTTYPE \
    DATATYPE  // Each main function should define this to make a result accessible after the MPC protocol has finished
#define FUNCTION Fixed_Point_Tutorial  // define your main entry point

template <typename Share>
void Fixed_Point_Tutorial(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;

    //---------------------------------//
    // Secret Sharing
    UINT_TYPE fixed_point_value = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(
        3.5f);  // Plaintext fixed point value with FRACTIONAL bits of precision, defined in config.h
    A fixed_point_secret;
    fixed_point_secret.template prepare_receive_from<P_0>(
        PROMOTE(fixed_point_value));  // After conversion, fixed point values can ban be treated as integers
    Share::communicate();
    fixed_point_secret.template complete_receive_from<P_0>();

    //---------------------------------//
    // Fixed Point Operations

    A result = fixed_point_secret + fixed_point_secret;  // Addition
    result = result - fixed_point_secret;                // Subtraction

    // Fixed Point Multiplication
    result = result.prepare_dot(fixed_point_secret);  // Local dot product computation
    result.mask_and_send_dot();                       // Mask and Send Dot Product, applies truncation
    Share::communicate();
    result.complete_mult();  // Complete multiplication with truncation

    // Multiplication with public fixed point values
    UINT_TYPE public_fixed_point_value =
        FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(2.5f);
    result = result.prepare_mult_public_fixed(
        public_fixed_point_value);  // While public integer values can be multiplied without communication, multiplying
                                    // a secret with a public fixed point values requires communication
    Share::communicate();
    result.complete_public_mult_fixed();  // Complete multiplication with a public fixed point value
                                          // Division with public values
    UINT_TYPE reciprocal = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(
        1 / 2.5f);  // simply invert the fixed point value and multiply
    result = result.prepare_mult_public_fixed(reciprocal);
    Share::communicate();
    result.complete_public_mult_fixed();

    //---------------------------------//
    // Reveal the result
    UINT_TYPE output_vectorized[vectorization_factor];
    float converted_output[vectorization_factor];
    DATATYPE output;
    result.prepare_reveal_to_all();
    Share::communicate();
    output = result.complete_reveal_to_all();
    unorthogonalize_arithmetic(&output, output_vectorized, 1);
    for (int i = 0; i < vectorization_factor; i++)
        converted_output[i] = FloatFixedConverter<INT_TYPE, UINT_TYPE, float, FRACTIONAL>::ufixed_to_float(
            output_vectorized[i]);  // Convert the fixed point value back to a floating point value
}
