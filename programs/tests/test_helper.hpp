#pragma once
#include "../../datatypes/Additive_Share.hpp"
template<typename FUNC>
void test_function(int &num_tests, int &num_passed, std::string name, FUNC f)
{
    print_online("Testing " + name + " ...");
    num_tests++;
    if(f())
    {
        num_passed++;
        print_online(name + " Passed!")
    }
    else
    {
        print_online(name + " Failed!")
    }
}

void print_stats(int num_tests, int num_passed)
{
    print_online("Passed " + std::to_string(num_passed) + " out of " + std::to_string(num_tests) + " tests.");
}

void print_compare(int expected, int got)
{
    print_online("Expected: " + std::to_string(expected) + " Got: " + std::to_string(got));
}

void print_compare(float expected, float got, float epsilon)
{
    print_online("Expected: " + std::to_string(expected) + " Got: " + std::to_string(got) + " Epsilon: " + std::to_string(epsilon));
}

