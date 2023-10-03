#pragma once
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <chrono>
#include <bitset>
#include <cstring>
#include "../../protocols/Protocols.h"
#include "../../protocols/XOR_Share.hpp"
#include "../../protocols/Additive_Share.hpp"
#define RESULTTYPE DATATYPE
#define COMMUNICATION_ROUNDS 1000

#if FUNCTION_IDENTIFIER < 4
#define FUNCTION MULT_BENCH
#else
#define FUNCTION MULT_BENCH_COMMUNICATION_ROUNDS
#endif

template<typename Share>
void MULT_BENCH(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    auto a = new S[NUM_INPUTS];
    auto b = new S[NUM_INPUTS];
    auto c = new S[NUM_INPUTS];
    Share::communicate(); // dummy round
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        c[i] = a[i] * b[i];
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        c[i].complete_mult();
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
using S = Additive_Share<DATATYPE, Share>;
auto a = new S[NUM_INPUTS];
auto b = new S[NUM_INPUTS];
auto c = new S[NUM_INPUTS];
Share::communicate(); // dummy communication round to simulate secret sharing

for(int j = 0; j < COMMUNICATION_ROUNDS; j++) {

for (int s = 0; s < loop_num; s++) {
    int i = s+j*loop_num;
    c[i] = a[i] * b[i];
}



Share::communicate();

for (int s = 0; s < loop_num; s++) {
    int i = s+j*loop_num;
    c[i].complete_mult();
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
 
