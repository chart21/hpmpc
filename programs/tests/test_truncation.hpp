#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/exact_truncation.hpp"
#include "../functions/prob_truncation.hpp"
#include "test_helper.hpp"

#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION test_truncation
#endif
#define TEST_PROB_TRUNC 1                // [a] -> [a]^t
#define TEST_PROB_TRUNC_REDUCED_SLACK 0  // [a] -> [a]^t
#define TEST_EXACT_TRUNC 1               // [a] -> [a]^t
#define TEST_EXACT_TRUNC_OPT 0           // [a] -> [a]^t

#if TEST_PROB_TRUNC == 1
template <typename Share>
bool test_prob_truncation()
{
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;
    // initialize plaintext inputs
    UINT_TYPE a[vectorization_factor];
    const int num = 20;
    float fa[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        fa[i] = num + i + float(num + i) / 100;
        a[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(fa[i], 3);
    }
    DATATYPE vectorized_input_a;
    orthogonalize_arithmetic(a, &vectorized_input_a, 1);

    // secret sharing
    A share_a;

    share_a.template prepare_receive_from<P_0>(vectorized_input_a);
    Share::communicate();
    share_a.template complete_receive_from<P_0>();

    // truncation
    trunc_pr_in_place(&share_a, 1);
    // reveal
    DATATYPE vecotrized_output;
    share_a.prepare_reveal_to_all();
    Share::communicate();
    vecotrized_output = share_a.complete_reveal_to_all();

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
        print_compare(fa[i] / (UINT_TYPE(1) << FRACTIONAL), output_float[i], epsilon);
        if (output_float[i] - fa[i] / (UINT_TYPE(1) << FRACTIONAL) > epsilon ||
            output_float[i] - fa[i] / (UINT_TYPE(1) << FRACTIONAL) < -epsilon)
        {
            return false;
        }
    }
    return true;
}

#endif

#if TEST_PROB_TRUNC_REDUCED_SLACK == 1
template <typename Share>
bool test_prob_reduced_slack_truncation()
{
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;
    // initialize plaintext inputs
    UINT_TYPE a[vectorization_factor];
    const int num = -123413;
    float fa[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        fa[i] = num + i + float(num + i) / 100;
        a[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(fa[i]);
    }
    DATATYPE vectorized_input_a;
    orthogonalize_arithmetic(a, &vectorized_input_a, 1);

    // secret sharing
    A share_a;

    share_a.template prepare_receive_from<P_0>(vectorized_input_a);
    Share::communicate();
    share_a.template complete_receive_from<P_0>();

    // truncation
    trunc_2k_in_place(&share_a, 1);
    // reveal
    DATATYPE vecotrized_output;
    share_a.prepare_reveal_to_all();
    Share::communicate();
    vecotrized_output = share_a.complete_reveal_to_all();

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
        print_compare(fa[i] / (UINT_TYPE(1) << FRACTIONAL), output_float[i], epsilon);
        if (output_float[i] - fa[i] / (UINT_TYPE(1) << FRACTIONAL) > epsilon ||
            output_float[i] - fa[i] / (UINT_TYPE(1) << FRACTIONAL) < -epsilon)
        {
            return false;
        }
    }
    return true;
}

#endif

#if TEST_EXACT_TRUNC == 1
template <typename Share>
bool test_exact_truncation()
{
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;
    // initialize plaintext inputs
    UINT_TYPE a[vectorization_factor];
    const int num = -123413;
    float fa[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        fa[i] = num + i + float(num + i) / 100;
        a[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(fa[i]);
    }
    DATATYPE vectorized_input_a;
    orthogonalize_arithmetic(a, &vectorized_input_a, 1);

    // secret sharing
    A share_a;

    share_a.template prepare_receive_from<P_0>(vectorized_input_a);
    Share::communicate();
    share_a.template complete_receive_from<P_0>();

    trunc_exact_in_place(&share_a, 1);
    // reveal
    DATATYPE vecotrized_output;
    share_a.prepare_reveal_to_all();
    Share::communicate();
    vecotrized_output = share_a.complete_reveal_to_all();

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
        print_compare(fa[i] / (UINT_TYPE(1) << FRACTIONAL), output_float[i], epsilon);
        if (output_float[i] - fa[i] / (UINT_TYPE(1) << FRACTIONAL) > epsilon ||
            output_float[i] - fa[i] / (UINT_TYPE(1) << FRACTIONAL) < -epsilon)
        {
            return false;
        }
    }
    return true;
}

#endif

#if TEST_EXACT_TRUNC_OPT == 1
template <typename Share>
bool test_opt_exact_truncation()
{
    using A = Additive_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;
    // initialize plaintext inputs
    UINT_TYPE a[vectorization_factor];
    const int num = -123413;
    float fa[vectorization_factor];
    for (int i = 0; i < vectorization_factor; i++)
    {
        fa[i] = num + i + float(num + i) / 100;
        a[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(fa[i]);
    }
    DATATYPE vectorized_input_a;
    orthogonalize_arithmetic(a, &vectorized_input_a, 1);

    // secret sharing
    A share_a;

    share_a.template prepare_receive_from<P_0>(vectorized_input_a);
    Share::communicate();
    share_a.template complete_receive_from<P_0>();

    trunc_exact_opt_in_place(&share_a, 1);
    // reveal
    DATATYPE vecotrized_output;
    share_a.prepare_reveal_to_all();
    Share::communicate();
    vecotrized_output = share_a.complete_reveal_to_all();

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
        print_compare(fa[i] / (UINT_TYPE(1) << FRACTIONAL), output_float[i], epsilon);
        if (output_float[i] - fa[i] / (UINT_TYPE(1) << FRACTIONAL) > epsilon ||
            output_float[i] - fa[i] / (UINT_TYPE(1) << FRACTIONAL) < -epsilon)
        {
            return false;
        }
    }
    return true;
}

#endif

template <typename Share>
bool test_truncation(DATATYPE* res)
{
    int num_tests = 0;
    int num_passed = 0;

#if TEST_PROB_TRUNC == 1
    test_function(num_tests, num_passed, "Probabilistic Truncation", test_prob_truncation<Share>);
#endif

#if TEST_PROB_TRUNC_REDUCED_SLACK == 1
    test_function(num_tests,
                  num_passed,
                  "Probabilistic Truncation with Reduced Slack",
                  test_prob_reduced_slack_truncation<Share>);
#endif

#if TEST_EXACT_TRUNC == 1
    test_function(num_tests, num_passed, "Exact Truncation", test_exact_truncation<Share>);
#endif

#if TEST_EXACT_TRUNC_OPT == 1
    test_function(num_tests, num_passed, "Optimized Exact Truncation", test_opt_exact_truncation<Share>);
#endif

    print_stats(num_tests, num_passed);
    if (num_tests == num_passed)
    {
        return true;
    }
    return false;
}
