#pragma once
#define FUNCTION test_all
#include "test_basic_primitives.hpp"
#include "test_comparisons.hpp"
#include "test_fixed_point_arithmetic.hpp"
#include "test_mat_mul.hpp"
#include "test_multi_input.hpp"
#include "test_truncation.hpp"

template <typename Share>
void test_all(DATATYPE* res)
{
    std::vector<std::string> test_names = {"Basic Primitives",
                                           "Comparisons",
                                           "Fixed Point Arithmetic",
                                           "Matrix Multiplication",
                                           "Multi Input Operations",
                                           "Truncation"};
    std::vector<bool> test_results;
    test_results.push_back(test_basic_primitives<Share>(res));
    test_results.push_back(test_comparisons<Share>(res));
    test_results.push_back(test_fixed_point_arithmetic<Share>(res));
    test_results.push_back(test_mat_mul<Share>(res));
    test_results.push_back(test_multi_input<Share>(res));
    test_results.push_back(test_truncation<Share>(res));
    for (int i = 0; i < test_results.size(); i++)
    {
        if (test_results[i])
        {
            print_online("All tests in the category " + test_names[i] + " passed!");
        }
        else
        {
            print_online("Some tests in the category " + test_names[i] + " failed!");
        }
    }
}
