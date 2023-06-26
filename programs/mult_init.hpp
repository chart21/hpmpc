#pragma once



#if FUNCTION_IDENTIFIER == 2 || FUNCTION_IDENTIFIER == 5 || FUNCTION_IDENTIFIER == 6 || FUNCTION_IDENTIFIER == 8
#define FUNCTION AND_BENCH_1
#include "functions/mult.hpp"
#else
#define FUNCTION AND_BENCH_COMMUNICATION_ROUNDS
#include "functions/mult_communication_rounds.hpp"
#endif



void generateElements()
{



}




