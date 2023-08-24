#pragma once
#include "functions/mat_mul.hpp"
void generateElements()
{
#if FUNCTION_IDENTIFIER < 12 || FUNCTION_IDENTIFIER == 18
        auto input = NEW(UINT_TYPE[NUM_INPUTS*2][DATTYPE]);
        player_input = NEW(DATATYPE[NUM_INPUTS*2*BITLENGTH]);
        /* for(int i = 0; i < NUM_INPUTS; ++i) { */
        for(int j = 0; j < DATTYPE; ++j) {
            #if FRACTIONAL > 0
            if(PARTY == 0) {
                double a = 1.6;
                double b = -2.2;
                double c = 3.3;
                double d = 4.11;
                input[0][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(a);
                input[1][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(b);
                input[2][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(c);
                input[3][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(d);
            } else {
                double a = 2.6;
                double b = 0.01;
                double c = 1.2;
                double d = 0.11;
                input[0][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(a);
                input[1][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(b);
                input[2][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(c);
                input[3][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(d);
            }
            #else
            if(PARTY == 0) {
                input[0][j] = 1;
                input[1][j] = 2;
                input[2][j] = 3;
                input[3][j] = 4;
            } else {
                input[0][j] = 2;
                input[1][j] = 0;
                input[2][j] = 1;
                input[3][j] = 3;
            }
#endif
        }
        /* } */
    #if PRINT == 1
        std::cout << PARTY << " input: ";
        for(int i = 0; i < NUM_INPUTS*2; ++i) {
    for(int j = 0; j < DATTYPE; j++)
    {    
        #if FRACTIONAL > 0
        std::cout << fixedToFloat<double, UINT_TYPE, FRACTIONAL>(input[i][j]) << " ";
        #else
        std::cout << input[i][j] << " ";
        #endif
   
    std::cout << std::endl;
    }
    std::cout << std::endl;
        }
    #endif
        
    /* orthogonalize_boolean(input, player_input); */
    for(int i = 0; i < NUM_INPUTS*2; ++i) {
    /* orthogonalize_boolean(input[i], player_input + BITLENGTH*i); */
        orthogonalize_arithmetic(input[i], player_input + BITLENGTH*i);
    }
#endif
}

void print_result(DATATYPE* var) 
{
}

