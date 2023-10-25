#pragma once
#include "functions/mat_mul.hpp"
#include <cstdlib>
#include <ctime>

void generateElements()
{
#if FUNCTION_IDENTIFIER == 28 //argmax
        /* UINT_TYPE fixed_init[] = {289383,634022,660336,120709,999999,908235,385644,229320,873269}; */
        auto input = NEW(UINT_TYPE[NUM_INPUTS][DATTYPE]);
        player_input = NEW(DATATYPE[NUM_INPUTS*BITLENGTH]);
        for(int i = 0; i < NUM_INPUTS; ++i) {
            for(int j = 0; j < DATTYPE; ++j) {
                
                /* input[i][j] = fixed_init[i]; */
                input[i][j] = rand() % 1000000;
            }
        }
        #if PARTY == 0
        srand(time(0));
        //set a random element in range 0,NUM_INPUTS to be the max
        UINT_TYPE max_index = rand() % NUM_INPUTS;
            std::cout << "Maxindex P" << PARTY << ": " << max_index << std::endl;
            for(int j = 0; j < DATTYPE; ++j) 
                input[max_index][j] = 10000000;
        #endif
        for(int i = 0; i < NUM_INPUTS; ++i) {
            orthogonalize_arithmetic(input[i], player_input + BITLENGTH*i);
#if PARTY == 0
                std::cout << "Player Inputs " << player_input[i*BITLENGTH] << std::endl;
#endif
        }
#endif

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
    #if PRINT == 1 && FUNCTION_IDENTIFIER == 18
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

