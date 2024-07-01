#pragma once
#include "../../protocols/Protocols.h"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/Additive_Share.hpp"
#define RESULTTYPE DATATYPE
#define COMMUNICATION_ROUNDS 1000

#if FUNCTION_IDENTIFIER == 1 || FUNCTION_IDENTIFIER == 2
#define FUNCTION MULT_BENCH
#elif FUNCTION_IDENTIFIER == 3 || FUNCTION_IDENTIFIER == 4
#define FUNCTION MULT_BENCH_COMMUNICATION_ROUNDS
#endif


template<typename Share>
void MULT_BENCH(DATATYPE* res)
{
#if FUNCTION_IDENTIFIER == 1
    using S = Additive_Share<DATATYPE, Share>;
#else
    using S = XOR_Share<DATATYPE, Share>;
#endif
    auto a = new S[NUM_INPUTS];
    auto b = new S[NUM_INPUTS];
    auto c = new S[NUM_INPUTS];
    Share::communicate(); // dummy round
    for(int i = 0; i < NUM_INPUTS; i++)
    {
#if FUNCTION_IDENTIFIER == 1
        c[i] = a[i].prepare_mult(b[i]);
#elif FUNCTION_IDENTIFIER == 2
        c[i] = a[i].prepare_and(b[i]);
#endif
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
#if FUNCTION_IDENTIFIER == 1
        c[i].complete_mult_without_trunc();
#elif FUNCTION_IDENTIFIER == 2
        c[i].complete_and();
#endif
    }
    Share::communicate();

    c[0].prepare_reveal_to_all();

    Share::communicate();

    *res = c[0].complete_reveal_to_all();

}

template<typename Share>
void MULT_BENCH_COMMUNICATION_ROUNDS (DATATYPE* res)
{
int loop_num = NUM_INPUTS/COMMUNICATION_ROUNDS;
#if FUNCTION_IDENTIFIER == 3
using S = Additive_Share<DATATYPE, Share>;
#else
using S = XOR_Share<DATATYPE, Share>;
#endif
auto a = new S[NUM_INPUTS];
auto b = new S[NUM_INPUTS];
auto c = new S[NUM_INPUTS];
Share::communicate(); // dummy communication round to simulate secret sharing

for(int j = 0; j < COMMUNICATION_ROUNDS; j++) {

for (int s = 0; s < loop_num; s++) {
    int i = s+j*loop_num;
#if FUNCTION_IDENTIFIER == 3
    c[i] = a[i].prepare_mult(b[i]);
#elif FUNCTION_IDENTIFIER == 4
    c[i] = a[i].prepare_and(b[i]);
#endif
}



Share::communicate();

for (int s = 0; s < loop_num; s++) {
    int i = s+j*loop_num;
#if FUNCTION_IDENTIFIER == 3
    c[i].complete_mult_without_trunc();
#elif FUNCTION_IDENTIFIER == 4
    c[i].complete_and();
#endif
}
Share::communicate();

}

c[0].prepare_reveal_to_all();

Share::communicate();

*res = c[0].complete_reveal_to_all();

Share::communicate();

}

  
void print_result(DATATYPE result[]) 
{
}
 