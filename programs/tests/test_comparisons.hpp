#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/Relu.hpp"
#include "../functions/comparisons.hpp"
#include "../functions/exact_truncation.hpp"
#include "../functions/prob_truncation.hpp"
#include "test_helper.hpp"
#pragma once
#include "../functions/max_min.hpp"
#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION test_comparisons
#endif
#define TEST_EQZ 1      // [a] == 0 ? [1] : [0]
#define TEST_LTZ 1      // [a] < 0 ? [1] : [0]
#define TEST_MAX_MIN 0  // [A] -> [max(A)] [min(A)]
#define TEST_RELU 1     // [a] > 0 ? [a] : 0
#if TEST_EQZ == 1
template <typename Share>
bool test_EQZ()
{
    const int vectorization_factor = DATTYPE / BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;

    // set shares from public values
    A eqz_output[3];
    A eqz_input[] = {A(0), A(1), A(-1)};

    // EQZ
    pack_additive<0, BITLENGTH>(eqz_input, eqz_output, 3, EQZ<0, BITLENGTH, Share, DATATYPE>);  // EQZ

    // reveal
    alignas(sizeof(DATATYPE)) UINT_TYPE output[3][vectorization_factor];
    reveal_and_store(eqz_output, output, 3);

    // compare
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(1, output[0][i]);
        print_compare(0, output[1][i]);
        print_compare(0, output[2][i]);

        if (output[0][i] != 1 || output[1][i] != 0 || output[2][i] != 0)
        {
            return false;
        }
    }
    return true;
}

#endif

#if TEST_LTZ == 1
template <typename Share>
bool test_LTZ()
{
    const int vectorization_factor = DATTYPE / BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;

    // set shares from public values
    A ltz_output[3];
    A ltz_input[] = {A(0), A(1), A(-1)};

    // LTZ
    pack_additive<0, BITLENGTH>(ltz_input, ltz_output, 3, LTZ<0, BITLENGTH, Share, DATATYPE>);

    // reveal
    alignas(sizeof(DATATYPE)) UINT_TYPE output[3][vectorization_factor];
    reveal_and_store(ltz_output, output, 3);

    // compare
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(0, output[0][i]);
        print_compare(0, output[1][i]);
        print_compare(1, output[2][i]);

        if (output[0][i] != 0 || output[1][i] != 0 || output[2][i] != 1)
        {
            return false;
        }
    }
    return true;
}

#endif

#if TEST_MAX_MIN == 1
template <typename Share>
bool max_min_test()
{
    Share::communicate();
    const int len = 400;
    const int vectorization_factor = DATTYPE / BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;
    // two indepndent arrays of lenth len, we want to find the max and min for each of the two
    A a[len];
    A b[len];
    for (int i = 0; i < len; i++)
    {
        a[i] = A(i);
        b[i] = A(len - i);
    }
    A merged[2 * len];
    A argmax[len];
    A argmin[len];
    A max_val[2];
    A min_val[2];
    A argmax_merged[2 * len];
    A argmin_merged[2 * len];

    // merge two lists two achieve both computations in equal communication rounds
    for (int i = 0; i < len; i++)
    {
        merged[i] = a[i];
        merged[i + len] = b[i];
    }

    max_min_sint<0, BITLENGTH>(merged, len, max_val, 2, true);  // batch size of 2, want max of each independent array

    max_min_sint<0, BITLENGTH>(merged, len, min_val, 2, false);  // batch size of 2, want min of each independent array
    argmax_argmin_sint<0, BITLENGTH>(merged, len, argmax_merged, 2, true);
    argmax_argmin_sint<0, BITLENGTH>(merged, len, argmin_merged, 2, false);
    // Extract final results
    A result[8] = {max_val[0],
                   min_val[0],
                   argmax_merged[len - 1],
                   argmin_merged[0],
                   max_val[1],
                   min_val[1],
                   argmax_merged[len + 0],
                   argmin_merged[len + len - 1]};
    alignas(sizeof(DATATYPE)) UINT_TYPE output[8][vectorization_factor];
    reveal_and_store(result, output, 8);
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(len - 1, output[0][i]);
        print_compare(0, output[1][i]);
        print_compare(1, output[2][i]);
        print_compare(1, output[3][i]);
        print_compare(len, output[4][i]);
        print_compare(1, output[5][i]);
        print_compare(1, output[6][i]);
        print_compare(1, output[7][i]);
        if (output[0][i] != len - 1 || output[1][i] != 0 || output[2][i] != 1 || output[3][i] != 1 ||
            output[4][i] != len || output[5][i] != 1 || output[6][i] != 1 || output[7][i] != 1)
        {
            return false;
        }
    }
    return true;
}
#endif

#if TEST_RELU == 1
template <typename Share>
bool test_RELU()
{
    const int vectorization_factor = DATTYPE / BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;

    // set shares from public values
    A relu_output[3];
    A relu_input[] = {A(100), A(11), A(-12)};

    // ReLU
    RELU<0, BITLENGTH, Share, DATATYPE>(relu_input, relu_input + 3, relu_output);
    // reveal
    alignas(sizeof(DATATYPE)) UINT_TYPE output[3][vectorization_factor];
    reveal_and_store(relu_output, output, 3);

    // compare
    for (int i = 0; i < vectorization_factor; i++)
    {
        print_compare(100, output[0][i]);
        print_compare(11, output[1][i]);
        print_compare(0, output[2][i]);

        if (output[0][i] != 100 || output[1][i] != 11 || output[2][i] != 0)
        {
            return false;
        }
    }
    return true;
}
#endif

template <typename Share>
bool test_comparisons(DATATYPE* res)
{
    int num_tests = 0;
    int num_passed = 0;

#if TEST_EQZ == 1
    test_function(num_tests, num_passed, "EQZ", test_EQZ<Share>);
#endif

#if TEST_LTZ == 1
    test_function(num_tests, num_passed, "LTZ", test_LTZ<Share>);
#endif

#if TEST_MAX_MIN == 1
    test_function(num_tests, num_passed, "MAX_MIN", max_min_test<Share>);
#endif

#if TEST_RELU == 1
    test_function(num_tests, num_passed, "RELU", test_RELU<Share>);
#endif

    print_stats(num_tests, num_passed);
    if (num_tests == num_passed)
    {
        return true;
    }
    return false;
}
