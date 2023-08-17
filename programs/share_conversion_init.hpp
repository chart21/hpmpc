#pragma once
#include "functions/share_conversion.hpp"
void generateElements()
{
        auto input = new DATATYPE[NUM_INPUTS][DATTYPE];
        player_input = new DATATYPE[NUM_INPUTS*BITLENGTH];
        for(int i = 0; i < NUM_INPUTS; ++i) {
        for(int j = 0; j < DATTYPE; ++j) {
            input[i][j] = j;
            if(i > 0)
                input[i][j] += 1;
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
    {    std::cout << std::bitset<sizeof(uint64_t)*8>(input[i][j]);
    std::cout << std::endl;
    }
    std::cout << std::endl;
        }
    #endif
        
    /* orthogonalize_boolean(input, player_input); */
    for(int i = 0; i < NUM_INPUTS; ++i) {
    orthogonalize_boolean(input[i], player_input + BITLENGTH*i);
    }
}

void print_result(DATATYPE* var) 
{
}

