#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/GEMM.hpp"
#include "test_helper.hpp"

#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION test_mat_mul
#endif
#define TEST_DOT_PRODUCT_INT 1    // [A] x [B] = [c]
#define TEST_DOT_PRODUCT_FIXED 1  // [A] x [B] = [c]
#define TEST_MAT_MULT_FIXED 1     // [A] x [B] = [C]

#if TEST_DOT_PRODUCT_INT == 1
template <typename Share>
bool dot_product_int_test()
{
    Share::communicate();
    const int vectorization_factor = DATTYPE / BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;
    const int l = 10;
    A X[10] = {A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8), A(9), A(10)};
    A W[10] = {A(1), A(2), A(3), A(4), A(5), A(6), A(7), A(8), A(9), A(10)};
    A Y[1] = {A(0)};
    UINT_TYPE x[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    UINT_TYPE w[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    UINT_TYPE y = 0;
    for (int i = 0; i < l; i++)
    {
        Y[0] += X[i].prepare_dot(W[i]);
    }
    Y[0].mask_and_send_dot_without_trunc();
    Share::communicate();
    Y[0].complete_mult_without_trunc();

    // Verification
    for (int i = 0; i < l; i++)
    {
        y += x[i] * w[i];
    }
    alignas(sizeof(DATATYPE)) UINT_TYPE output[1][vectorization_factor];
    reveal_and_store(Y, output, 1);
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(y, output[0][i]);
        if (y != output[0][i])
        {
            return false;
        }
    }
    return true;
}
#endif

#if TEST_DOT_PRODUCT_FIXED == 1
template <typename Share>
bool dot_product_fixed_test()
{
    Share::communicate();
    const int vectorization_factor = DATTYPE / BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;
    const int l = 10;
    A X[10];
    A W[10];
    for (int i = 0; i < l; i++)
    {
        X[i] = FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(10 + i +
                                                                                                float(10 + i) / 100);
        W[i] = X[i];
    }
    A Y[1] = {A(0)};
    float x[10] = {10.f + float(10) / 100,
                   11.f + float(11) / 100,
                   12.f + float(12) / 100,
                   13.f + float(13) / 100,
                   14.f + float(14) / 100,
                   15.f + float(15) / 100,
                   16.f + float(16) / 100,
                   17.f + float(17) / 100,
                   18.f + float(18) / 100,
                   19.f + float(19) / 100};
    float w[10] = {10.f + float(10) / 100,
                   11.f + float(11) / 100,
                   12.f + float(12) / 100,
                   13.f + float(13) / 100,
                   14.f + float(14) / 100,
                   15.f + float(15) / 100,
                   16.f + float(16) / 100,
                   17.f + float(17) / 100,
                   18.f + float(18) / 100,
                   19.f + float(19) / 100};
    float y = 0;
    for (int i = 0; i < l; i++)
    {
        Y[0] += X[i].prepare_dot(W[i]);
    }
    Y[0].mask_and_send_dot();
    Share::communicate();
    Y[0].complete_mult();

    // Verification
    for (int i = 0; i < l; i++)
    {
        y += x[i] * w[i];
    }
    alignas(sizeof(DATATYPE)) UINT_TYPE output[1][vectorization_factor];
    reveal_and_store(Y, output, 1);
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(
            y, FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(output[0][i]), epsilon);
        if (y - FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(output[0][i]) > epsilon ||
            y - FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(output[0][i]) < -epsilon)
        {
            return false;
        }
    }
    return true;
}
#endif

#if TEST_MAT_MULT_FIXED == 1
template <typename Share>
bool mat_mul_fixed_test()
{
    Share::communicate();
    const int vectorization_factor = DATTYPE / BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;
    const int m = 3;
    const int k = 2;
    const int n = 1;
    float x[m * k] = {3.2, 5.4, 7.6, 11.8, 2.1, 4.3};  // initialize plaintext inputs for comparison
    float w[k * n] = {2.1, 4.3};
    float y[m * n] = {0};
    auto X = new A[m * k]{A(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(3.2)),
                          A(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(5.4)),
                          A(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(7.6)),
                          A(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(11.8)),
                          A(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(2.1)),
                          A(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(4.3))};
    auto W = new A[k * n]{A(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(2.1)),
                          A(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(4.3))};
    auto Y = new A[m * n]{A(0), A(0), A(0)};

    Share::communicate();

    prepare_GEMM(X, W, Y, m, n, k, false);
    Share::communicate();
    complete_GEMM(Y, m * n);

    // Naive Mat Mul for correctness test
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int q = 0; q < k; q++)
            {
                y[i * n + j] += x[i * k + q] * w[q * n + j];
            }
        }
    }
    alignas(sizeof(DATATYPE)) UINT_TYPE output[m * n][vectorization_factor];
    reveal_and_store(Y, output, m * n);
    for (int i = 0; i < m * n; i++)
    {
        for (int j = 0; j < vectorization_factor; j++)
        {
            print_compare(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(output[i][j]),
                          y[i],
                          epsilon);
            if (FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(output[i][j]) - y[i] >
                    epsilon ||
                FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::ufixed_to_float(output[i][j]) - y[i] <
                    -epsilon)
            {
                delete[] X;
                delete[] W;
                delete[] Y;
                return false;
            }
        }
    }
    delete[] X;
    delete[] W;
    delete[] Y;
    return true;
}

#endif

/* template<typename Share> */
/* void test_truncation(DATATYPE *res) */
/* { */
/*     int num_tests = 0; */
/*     int num_passed = 0; */

/*     #if TEST_PROB_TRUNC == 1 */
/*     test_function(num_tests, num_passed, "Probabilistic Truncation", test_prob_truncation<Share>); */
/*     #endif */

/* #if TEST_PROB_TRUNC_REDUCED_SLACK == 1 */
/*     test_function(num_tests, num_passed, "Probabilistic Truncation with Reduced Slack",
 * test_prob_reduced_slack_truncation<Share>); */
/* #endif */

/* #if TEST_EXACT_TRUNC == 1 */
/*     test_function(num_tests, num_passed, "Exact Truncation", test_exact_truncation<Share>); */
/* #endif */

/*     print_stats(num_tests, num_passed); */
/* } */

template <typename Share>
bool test_mat_mul(DATATYPE* res)
{
    int num_tests = 0;
    int num_passed = 0;

#if TEST_DOT_PRODUCT_INT == 1
    test_function(num_tests, num_passed, "Dot Product Int", dot_product_int_test<Share>);
#endif

#if TEST_DOT_PRODUCT_FIXED == 1
    test_function(num_tests, num_passed, "Dot Product Fixed", dot_product_fixed_test<Share>);
#endif

#if TEST_MAT_MULT_FIXED == 1
    test_function(num_tests, num_passed, "Matrix Multiplication", mat_mul_fixed_test<Share>);
#endif

    print_stats(num_tests, num_passed);
    if (num_tests == num_passed)
    {
        return true;
    }
    return false;
}
