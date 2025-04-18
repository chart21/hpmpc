#pragma once
#include "test_helper.hpp"
#ifndef FUNCTION
#define FUNCTION test_basic_primitives
#endif
#define RESULTTYPE DATATYPE
#define TEST_SECRET_SHARING 1      // a -> [a]
#define TEST_ADD_MULT_CONSTANTS 1  // b -> [b], [a] + [b], [a] * c
#define TEST_MULTIPLICATION 1      // [a] * [b]

#if TEST_SECRET_SHARING == 1
template <typename Share, int party_id>
bool test_secret_sharing_and_revealing()
{
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;

    // initialize plaintext inputs
    UINT_TYPE inputs[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        inputs[i] = 20 + i;
    }
    DATATYPE vectorized_input;
    orthogonalize_arithmetic(inputs, &vectorized_input, 1);

    // secret sharing
    A share;
    share.template prepare_receive_from<party_id>(vectorized_input);
    Share::communicate();
    share.template complete_receive_from<party_id>();

    // reveal
    DATATYPE vecotrized_output;
    share.prepare_reveal_to_all();
    Share::communicate();
    vecotrized_output = share.complete_reveal_to_all();

    // unorthogonalize
    UINT_TYPE output[vectorization_factor];
    unorthogonalize_arithmetic(&vecotrized_output, output, 1);

    // compare
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(inputs[i], output[i]);
        if (output[i] != inputs[i])
        {
            return false;
        }
    }
    return true;
}
#endif

#if TEST_ADD_MULT_CONSTANTS == 1
template <typename Share>
bool test_add_mult_consants()
{
    const int constant = 5;
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;

    // initialize plaintext inputs
    UINT_TYPE inputs[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        inputs[i] = 20 + i;
    }
    DATATYPE vectorized_input;
    orthogonalize_arithmetic(inputs, &vectorized_input, 1);

    // secret sharing
    A share;
    share.template prepare_receive_from<P_0>(vectorized_input);
    Share::communicate();
    share.template complete_receive_from<P_0>();

    // constant addition
    share += A(constant);
    share = share.mult_public(constant);

    // reveal
    DATATYPE vecotrized_output;
    share.prepare_reveal_to_all();
    Share::communicate();
    vecotrized_output = share.complete_reveal_to_all();

    // unorthogonalize
    UINT_TYPE output[vectorization_factor];
    unorthogonalize_arithmetic(&vecotrized_output, output, 1);

    // compare
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare((inputs[i] + constant) * constant, output[i]);
        if (output[i] != (inputs[i] + constant) * constant)
        {
            return false;
        }
    }
    return true;
}
#endif

#if TEST_MULTIPLICATION == 1
template <typename Share>
bool test_multiplication()
{
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;

    // initialize plaintext inputs
    UINT_TYPE a[vectorization_factor];
    UINT_TYPE b[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        a[i] = 20 + i;
        b[i] = 30 + i;
    }
    DATATYPE vectorized_input_a;
    DATATYPE vectorized_input_b;
    orthogonalize_arithmetic(a, &vectorized_input_a, 1);
    orthogonalize_arithmetic(b, &vectorized_input_b, 1);

    // secret sharing
    A share_a;
    A share_b;
    A share_c;

    share_a.template prepare_receive_from<P_0>(vectorized_input_a);
    share_b.template prepare_receive_from<P_1>(vectorized_input_b);
    Share::communicate();
    share_a.template complete_receive_from<P_0>();
    share_b.template complete_receive_from<P_1>();

    // multiplication
    share_c = share_a.prepare_mult(share_b);
    Share::communicate();
    share_c.complete_mult_without_trunc();

    // reveal
    DATATYPE vecotrized_output;
    share_c.prepare_reveal_to_all();
    Share::communicate();
    vecotrized_output = share_c.complete_reveal_to_all();

    // unorthogonalize
    UINT_TYPE output[vectorization_factor];
    unorthogonalize_arithmetic(&vecotrized_output, output, 1);

    // compare
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(a[i] * b[i], output[i]);
        if (output[i] != (a[i] * b[i]))
        {
            return false;
        }
    }
    return true;
}
#endif

template <typename Share>
bool test_basic_primitives(DATATYPE* res)
{
    int num_tests = 0;
    int num_passed = 0;
#if TEST_SECRET_SHARING == 1
    test_function(num_tests,
                  num_passed,
                  "Secret Sharing and Revealing, Input from P_0",
                  test_secret_sharing_and_revealing<Share, P_0>);
    test_function(num_tests,
                  num_passed,
                  "Secret Sharing and Revealing, Input from P_1",
                  test_secret_sharing_and_revealing<Share, P_1>);
#if num_players > 2
    test_function(num_tests,
                  num_passed,
                  "Secret Sharing and Revealing P_2, Input from P_2",
                  test_secret_sharing_and_revealing<Share, P_2>);
#endif
#if num_players > 3
    test_function(num_tests,
                  num_passed,
                  "Secret Sharing and Revealing P_3, Input from P_3",
                  test_secret_sharing_and_revealing<Share, P_3>);
#endif
#endif

#if TEST_ADD_MULT_CONSTANTS == 1
    test_function(num_tests, num_passed, "Addition and Multiplication with Constants", test_add_mult_consants<Share>);
#endif

#if TEST_MULTIPLICATION == 1
    test_function(num_tests, num_passed, "Multiplication", test_multiplication<Share>);
#endif

    print_stats(num_tests, num_passed);
    if (num_tests == num_passed)
    {
        return true;
    }
    return false;
}
