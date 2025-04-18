#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/Relu.hpp"
#include "../functions/exact_truncation.hpp"
#include "../functions/prob_truncation.hpp"
#include "test_helper.hpp"
#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION test_fixed_point_arithmetic
#endif
#define TEST_FIXED_MULTIPLICATION 1         // [a] * c
#define TEST_PUBLIC_FIXED_MULTIPLICATION 1  // [a] * [b]
#define TEST_PUBLIC_FIXED_DIVISION 1        // [a] / c
#define TEST_NEWTON_RAPHSON_DIVISION 1      // [a] / [b]

#if TEST_PUBLIC_FIXED_DIVISION == 1
template <typename Share>
bool public_division()
{
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;
    // initialize plaintext inputs
    UINT_TYPE a[vectorization_factor];
    UINT_TYPE b[vectorization_factor];
    float fa[vectorization_factor];
    float fb[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        fa[i] = 20 + i + float(20 + i) / 100;
        fb[i] = 5;
        a[i] = FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(fa[i]);
        b[i] = FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1 / fb[i]);
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
    Share::communicate();
    share_a.template complete_receive_from<P_0>();

    // multiplication
    share_c = share_a.prepare_mult_public_fixed_dat(vectorized_input_b);
    Share::communicate();
    share_c.complete_public_mult_fixed();

    // reveal
    DATATYPE vecotrized_output;
    share_c.prepare_reveal_to_all();
    Share::communicate();
    vecotrized_output = share_c.complete_reveal_to_all();

    // unorthogonalize
    UINT_TYPE output[vectorization_factor];
    unorthogonalize_arithmetic(&vecotrized_output, output, 1);
    float output_float[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        output_float[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(output[i]);
    }

    // compare
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(fa[i] / fb[i], output_float[i], epsilon);
        if (output_float[i] - fa[i] / fb[i] > epsilon || output_float[i] - fa[i] / fb[i] < -epsilon)
        {
            return false;
        }
    }
    return true;
}

#endif

#if TEST_NEWTON_RAPHSON_DIVISION == 1
template <typename Share>
bool division()
{
    Share::communicate();
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;
    // initialize plaintext inputs
    UINT_TYPE a[vectorization_factor];
    UINT_TYPE b[vectorization_factor];
    float fa[vectorization_factor];
    float fb[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        fa[i] = 20 + i + float(20 + i) / 100;
        fb[i] = 5;
        a[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(fa[i]);
        b[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(fb[i]);
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
    Share::communicate();
    share_a.template complete_receive_from<P_0>();

    share_b.template prepare_receive_from<P_1>(vectorized_input_b);
    Share::communicate();
    share_b.template complete_receive_from<P_1>();

    // multiplication
    // y0(x) = 3e^(0.5−x) + 0.003 -> initial guess
    // initial guess
    share_c =
        FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(3 * exp(0.5 - float(5)) + 0.003);
    // Newpthon Raphson formula 1/x = limn→∞ yn = y_n−1(2 − xyn−1)
    for (int i = 0; i < 10; i++)
    {
        auto res1 = share_b.prepare_dot(share_c);
        auto res2 = share_c + share_c;
        res1.mask_and_send_dot();
        Share::communicate();
        res1.complete_mult();
        auto res3 = share_c.prepare_dot(res1);
        res3.mask_and_send_dot();
        Share::communicate();
        res3.complete_mult();
        share_c = res2 - res3;
    }
    share_c = share_a.prepare_dot(share_c);  // a * 1/b
    share_c.mask_and_send_dot();
    Share::communicate();
    share_c.complete_mult();

    // reveal
    DATATYPE vecotrized_output;
    share_c.prepare_reveal_to_all();
    Share::communicate();
    vecotrized_output = share_c.complete_reveal_to_all();

    // unorthogonalize
    UINT_TYPE output[vectorization_factor];
    unorthogonalize_arithmetic(&vecotrized_output, output, 1);
    float output_float[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        output_float[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(output[i]);
    }

    // compare
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(fa[i] / fb[i], output_float[i], epsilon);
        if (output_float[i] - fa[i] / fb[i] > epsilon || output_float[i] - fa[i] / fb[i] < -epsilon)
        {
            return false;
        }
    }
    return true;
}

#endif

#if TEST_PUBLIC_FIXED_MULTIPLICATION == 1
template <typename Share>
bool pulbic_fixed_multiplication()
{
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;
    // initialize plaintext inputs
    UINT_TYPE a[vectorization_factor];
    UINT_TYPE b[vectorization_factor];
    float fa[vectorization_factor];
    float fb[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        fa[i] = 20 + i + float(20 + i) / 100;
        fb[i] = 20 + float(20 + i) / 100;
        a[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(fa[i]);
        b[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(fb[i]);
    }
    DATATYPE vectorized_input_a;
    DATATYPE vectorized_constant_b;
    orthogonalize_arithmetic(a, &vectorized_input_a, 1);
    orthogonalize_arithmetic(b, &vectorized_constant_b, 1);

    // secret sharing
    A share_a;
    A share_b;
    A share_c;

    share_a.template prepare_receive_from<P_0>(vectorized_input_a);
    Share::communicate();
    share_a.template complete_receive_from<P_0>();

    // multiplication with reciprocal
    share_c = share_a.prepare_mult_public_fixed_dat(vectorized_constant_b);
    Share::communicate();
    share_c.complete_public_mult_fixed();

    // reveal
    DATATYPE vecotrized_output;
    share_c.prepare_reveal_to_all();
    Share::communicate();
    vecotrized_output = share_c.complete_reveal_to_all();

    // unorthogonalize
    UINT_TYPE output[vectorization_factor];
    unorthogonalize_arithmetic(&vecotrized_output, output, 1);
    float output_float[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        output_float[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(output[i]);
    }

    // compare
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(fa[i] * fb[i], output_float[i], epsilon);
        if (output_float[i] - fa[i] * fb[i] > epsilon || output_float[i] - fa[i] * fb[i] < -epsilon)
        {
            return false;
        }
    }
    return true;
}
#endif

#if TEST_FIXED_MULTIPLICATION == 1
template <typename Share>
bool test_fixed_multiplication()
{
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;
    // initialize plaintext inputs
    UINT_TYPE a[vectorization_factor];
    UINT_TYPE b[vectorization_factor];
    float fa[vectorization_factor];
    float fb[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        fa[i] = 20 + i + float(20 + i) / 100;
        fb[i] = -20 + float(20 + i) / 100;
        a[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(fa[i]);
        b[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(fb[i]);
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
    share_c = share_a.prepare_dot(share_b);
#if TRUNC_APPROACH == 0 && TRUNC_DELAYED == 0
    share_c.mask_and_send_dot();
#else
    share_c.mask_and_send_dot_without_trunc();
#endif
    Share::communicate();
#if TRUNC_APPROACH == 0 && TRUNC_DELAYED == 0
    share_c.complete_mult();
#else
    share_c.complete_mult_without_trunc();
    Share::communicate();
#if TRUNC_APPROACH == 0
    trunc_pr_in_place(&share_c, 1);
#elif TRUNC_APPROACH == 1
    trunc_2k_in_place(&share_c, 1, false);
#elif TRUNC_APPROACH == 2
    trunc_exact_in_place(&share_c, 1);
#endif
#endif

    // reveal
    DATATYPE vecotrized_output;
    share_c.prepare_reveal_to_all();
    Share::communicate();
    vecotrized_output = share_c.complete_reveal_to_all();

    // unorthogonalize
    UINT_TYPE output[vectorization_factor];
    unorthogonalize_arithmetic(&vecotrized_output, output, 1);
    float output_float[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        output_float[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(output[i]);
    }

    // compare
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(fa[i] * fb[i], output_float[i], epsilon);
        if (output_float[i] - fa[i] * fb[i] > epsilon || output_float[i] - fa[i] * fb[i] < -epsilon)
        {
            return false;
        }
    }
    return true;
}
#endif

template <typename Share>
bool test_fixed_point_arithmetic(DATATYPE* res)
{
    int num_tests = 0;
    int num_passed = 0;

#if TEST_FIXED_MULTIPLICATION == 1
    test_function(num_tests, num_passed, "Fixed Point Multiplication", test_fixed_multiplication<Share>);
#endif

#if TEST_PUBLIC_FIXED_MULTIPLICATION == 1
    test_function(
        num_tests, num_passed, "Fixed Point Multiplication with public constant", pulbic_fixed_multiplication<Share>);
#endif

#if TEST_PUBLIC_FIXED_DIVISION == 1
    test_function(num_tests, num_passed, "Fixed Point Division with public constant", public_division<Share>);
#endif

#if TEST_NEWTON_RAPHSON_DIVISION == 1
    test_function(num_tests, num_passed, "Fixed Point Division with secret shared share", division<Share>);
#endif

    print_stats(num_tests, num_passed);
    if (num_tests == num_passed)
    {
        return true;
    }
    return false;
}
