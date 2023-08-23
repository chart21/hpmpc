#pragma once
#include "functions/mat_mul.hpp"
void generateElements()
{
#if FUNCTION_IDENTIFIER < 12
        auto input = NEW(UINT_TYPE[NUM_INPUTS][DATTYPE]);
        player_input = NEW(DATATYPE[NUM_INPUTS*BITLENGTH]);
        /* for(int i = 0; i < NUM_INPUTS; ++i) { */
        for(int j = 0; j < DATTYPE; ++j) {
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
        }
        /* } */
    #if PRINT == 1
        std::cout << PARTY << " input: ";
        for(int i = 0; i < NUM_INPUTS; ++i) {
    for(int j = 0; j < DATTYPE; j++)
    {    std::cout <<  input[i][j] << " ";
   
    std::cout << std::endl;
    }
    std::cout << std::endl;
        }
    #endif
        
    /* orthogonalize_boolean(input, player_input); */
    for(int i = 0; i < NUM_INPUTS; ++i) {
    /* orthogonalize_boolean(input[i], player_input + BITLENGTH*i); */
        orthogonalize_arithmetic(input[i], player_input + BITLENGTH*i);
    }
#endif
}

void print_result(DATATYPE* var) 
{
}

