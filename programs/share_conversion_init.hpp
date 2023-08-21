#pragma once
#include "functions/share_conversion.hpp"
void generateElements()
{
        auto input = NEW(UINT_TYPE[NUM_INPUTS][DATTYPE]);
        player_input = NEW(DATATYPE[NUM_INPUTS*BITLENGTH]);
        for(int i = 0; i < NUM_INPUTS; ++i) {
        for(int j = 0; j < DATTYPE; ++j) {
            /* if(i % 2 == 0) */
            /*     input[i][j] = -j*(i+1); */
            /* else */
                input[i][j] = j;

            /* if (j % (i+1) == 0) */
            /*     input[i][j] = j; */
            /* else */
            /*     input[i][j] = -(12 + j); */
        }
        }
    #if PRINT == 1
        std::cout << PARTY << " input: ";
        for(int i = 0; i < NUM_INPUTS; ++i) {
    for(int j = 0; j < DATTYPE; j++)
    {    std::cout << std::bitset<sizeof(UINT_TYPE)*8>(input[i][j]);
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
}

void print_result(DATATYPE* var) 
{
}

