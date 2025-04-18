#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/GEMM.hpp"
#include "test_helper.hpp"

#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION test_multi_input
#endif
#define TEST_MULTIMULT 1  // [a] * [b] * [c] = [d]
#define TEST_MULTIDOT 1   // [A] x [B] x [C] = [d]

#if TEST_MULTIMULT == 1
template <typename Share>
bool mult34_test()
{
    Share::communicate();
    const int vectorization_factor = DATTYPE / BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;
    A inputsa[4] = {A(2), A(3), A(4), A(5)};
    A inputsb[6] = {A(3), A(4), A(5), A(6), A(7), A(8)};
    A result_mul3;
    A result_mul4;

    result_mul3 = inputsa[0].prepare_mult3(inputsa[1], inputsa[2]);
    result_mul4 = inputsa[0].prepare_mult4(inputsa[1], inputsa[2], inputsa[3]);
    Share::communicate();
    result_mul3.complete_mult3();
    result_mul4.complete_mult4();

    result_mul3 = result_mul3.prepare_mult3(inputsb[0], inputsb[1]);
    result_mul4 = result_mul4.prepare_mult4(inputsb[0], inputsb[1], inputsb[2]);
    Share::communicate();
    result_mul3.complete_mult3();
    result_mul4.complete_mult4();

    result_mul3 = result_mul3.prepare_mult3(inputsb[2], inputsb[3]);
    result_mul4 = result_mul4.prepare_mult4(inputsb[3], inputsb[4], inputsb[5]);
    Share::communicate();
    result_mul3.complete_mult3();
    result_mul4.complete_mult4();

    UINT_TYPE ver_result3 = 2 * 3 * 4 * 3 * 4 * 5 * 6;
    UINT_TYPE ver_result4 = 2 * 3 * 4 * 5 * 3 * 4 * 5 * 6 * 7 * 8;
    A result[2] = {result_mul3, result_mul4};
    alignas(sizeof(DATATYPE)) UINT_TYPE output[2][vectorization_factor];
    reveal_and_store(result, output, 2);
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(ver_result3, output[0][i]);
        print_compare(ver_result4, output[1][i]);
        if (ver_result3 != output[0][i] || ver_result4 != output[1][i])
        {
            return false;
        }
    }
    return true;
}
#endif

#if TEST_MULTIDOT == 1
template <typename Share>
bool dot234_test()
{
    Share::communicate();
    const int vectorization_factor = DATTYPE / BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;
    UINT_TYPE a[10] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    UINT_TYPE b[10] = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    A inputsa[10] = {A(a[0]), A(a[1]), A(a[2]), A(a[3]), A(a[4]), A(a[5]), A(a[6]), A(a[7]), A(a[8]), A(a[9])};
    A inputsb[10] = {A(b[0]), A(b[1]), A(b[2]), A(b[3]), A(b[4]), A(b[5]), A(b[6]), A(b[7]), A(b[8]), A(b[9])};
    auto result1 = inputsa[0].prepare_dot(inputsa[1]) + inputsa[2].prepare_dot3(inputsa[3], inputsa[4]) +
                   inputsa[5].prepare_dot4(inputsa[6], inputsa[7], inputsa[8]);
    auto result2 = inputsb[0].prepare_dot(inputsb[1]) + inputsb[2].prepare_dot3(inputsb[3], inputsb[4]) +
                   inputsb[5].prepare_dot4(inputsb[6], inputsb[7], inputsb[8]);
    result1.mask_and_send_dot_without_trunc();
    result2.mask_and_send_dot_without_trunc();
    Share::communicate();
    result1.complete_mult_without_trunc();
    result2.complete_mult_without_trunc();
    auto result = result1.prepare_dot4(result2, inputsa[9], inputsb[9]);
    result.mask_and_send_dot_without_trunc();
    Share::communicate();
    result.complete_mult_without_trunc();

    UINT_TYPE ver_result1 = a[0] * a[1] + a[2] * a[3] * a[4] + a[5] * a[6] * a[7] * a[8];
    UINT_TYPE ver_result2 = b[0] * b[1] + b[2] * b[3] * b[4] + b[5] * b[6] * b[7] * b[8];
    UINT_TYPE ver_result = ver_result1 * ver_result2 * a[9] * b[9];

    alignas(sizeof(DATATYPE)) UINT_TYPE output[1][vectorization_factor];
    reveal_and_store(&result, output, 1);
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(ver_result, output[0][i]);
        if (ver_result != output[0][i])
        {
            return false;
        }
    }
    return true;
}

#endif

template <typename Share>
bool test_multi_input(DATATYPE* res)
{
    int num_tests = 0;
    int num_passed = 0;

#if TEST_MULTIMULT == 1
    test_function(num_tests, num_passed, "Multi-input Multiplication", mult34_test<Share>);
#endif

#if TEST_MULTIDOT == 1
    test_function(num_tests, num_passed, "Multi-input Scalar Products", dot234_test<Share>);
#endif

    print_stats(num_tests, num_passed);
    if (num_tests == num_passed)
    {
        return true;
    }
    return false;
}
