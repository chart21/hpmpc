#pragma once
#include "functions/mat_mul.hpp"
#include "../utils/print.hpp"
#include <cstdlib>
#include <ctime>

void generateElements()
{
    /* for(int i = 0; i < 1000; ++i) { */
    /*     auto a = getRandomVal(PPREV); */
    /*     auto b = getRandomVal(PNEXT); */
/* #if PARTY == 0 */
    /*     print_result(&a); */
    /*     print_result(&b); */
/* #endif */
    /* } */
#if FUNCTION_IDENTIFIER == 28 //argmax
        /* UINT_TYPE fixed_init[] = {289383,634022,660336,120709,999999,908235,385644,229320,873269}; */
        auto input = NEW(UINT_TYPE[NUM_INPUTS][DATTYPE]);
        player_input = NEW(DATATYPE[NUM_INPUTS*BITLENGTH]);
        for(int i = 0; i < NUM_INPUTS; ++i) {
            for(int j = 0; j < DATTYPE; ++j) {
                
                /* input[i][j] = fixed_init[i]; */
                input[i][j] = rand() % 200 - 100;
            }
        }
        #if PARTY == 2 || PROTOCOL == 13
        srand(time(0));
        //set a random element in range 0,NUM_INPUTS to be the max
        UINT_TYPE max_index = rand() % NUM_INPUTS;
        UINT_TYPE min_index = rand() % NUM_INPUTS;
        while(max_index == min_index)
            min_index = rand() % NUM_INPUTS;
        std::cout << "Maxindex P" << PARTY << ": " << std::to_string(max_index) << std::endl;
        std::cout << "Minindex P" << PARTY << ": " << std::to_string(min_index) << std::endl;
            for(int j = 0; j < DATTYPE; ++j) 
            {
                input[max_index][j] = 200;
                input[min_index][j] = -200;
            }
        #endif
        for(int i = 0; i < NUM_INPUTS; ++i) {
            orthogonalize_arithmetic(input[i], player_input + BITLENGTH*i);
/* #if PARTY == 0 */
/*                 std::cout << "Player Inputs " << player_input[i*BITLENGTH] << std::endl; */
/* #endif */
        }
#endif

#if FUNCTION_IDENTIFIER < 12 || FUNCTION_IDENTIFIER == 18
        auto input = NEW(UINT_TYPE[NUM_INPUTS*NUM_INPUTS][DATTYPE]);
        player_input = NEW(DATATYPE[NUM_INPUTS*NUM_INPUTS*BITLENGTH]);
        /* for(int i = 0; i < NUM_INPUTS; ++i) { */
        for(int j = 0; j < DATTYPE; ++j) {
            #if FRACTIONAL > 0
            if(PARTY == 0) {
                double a = 1000.6;
                /* double b = -2.2; */
                double b = 2000.01;
                double c = 3000.3;
                double d = 4000.11;
                double e = 5000.11;
                double f = 6000.11;
                double g = 7000.11;
                double h = 8000.11;
                double i = 9000.11;
                input[0][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(a);
                input[1][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(b);
                input[2][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(c);
                input[3][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(d);
                input[4][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(e);
                input[5][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(f);
                input[6][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(g);
                input[7][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(h);
                input[8][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(i);
            } else {
                double a = 2.6;
                double b = 0.01;
                double c = 1.2;
                double d = 0.11;
                double e = 0.11;
                double f = 0.11;
                double g = 0.11;
                double h = 0.11;
                double i = 0.11;
                input[0][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(a);
                input[1][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(b);
                input[2][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(c);
                input[3][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(d);
                input[4][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(e);
                input[5][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(f);
                input[6][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(g);
                input[7][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(h);
                input[8][j] = floatToFixed<double, UINT_TYPE, FRACTIONAL>(i);

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
        for(int i = 0; i < NUM_INPUTS*NUM_INPUTS; ++i) {
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
    for(int i = 0; i < NUM_INPUTS*NUM_INPUTS; ++i) {
    /* orthogonalize_boolean(input[i], player_input + BITLENGTH*i); */
        orthogonalize_arithmetic(input[i], player_input + BITLENGTH*i);
    }
#endif

#if FUNCTION_IDENTIFIER == 29 || FUNCTION_IDENTIFIER == 30 // Multi input AND gates
        player_input = NEW(DATATYPE[NUM_INPUTS]);
        UINT_TYPE inputs[] = {1,2,3,5};
        for(int j = 0; j < NUM_INPUTS; j++)
        {   
            player_input[j] = PROMOTE(inputs[j]);
        }
#endif
#if FUNCTION_IDENTIFIER == 31 || FUNCTION_IDENTIFIER == 32 || FUNCTION_IDENTIFIER == 33 || FUNCTION_IDENTIFIER == 34 // Dot Product 234 test
        player_input = NEW(DATATYPE[NUM_INPUTS]);
        /* UINT_TYPE inputs[] = {1,2,3,5}; */
        UINT_TYPE input = 3;
        for(int j = 0; j < NUM_INPUTS; j++)
        {   
            player_input[j] = PROMOTE(input);
        }
#endif

#if FUNCTION_IDENTIFIER == 35
        auto input = NEW(UINT_TYPE[NUM_INPUTS][DATTYPE]);
        player_input = NEW(DATATYPE[NUM_INPUTS*BITLENGTH]);
        srand(time(0));
        for(int i = 0; i < NUM_INPUTS; ++i) {
            for(int j = 0; j < DATTYPE; ++j) {
                
                /* input[i][j] = fixed_init[i]; */
                input[i][j] = rand() % 200 - 100;
            }
#if PARTY == 0
            std::cout << "Input " << i << ": " << input[i][0] << std::endl;
#endif
        }
        
        for(int i = 0; i < NUM_INPUTS; ++i) {
            orthogonalize_arithmetic(input[i], player_input + BITLENGTH*i);
        }

#endif


}

void print_result(DATATYPE* var) 
{

}

