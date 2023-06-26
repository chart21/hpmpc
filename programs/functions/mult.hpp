#pragma once
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <chrono>
#include <bitset>
#include <cstring>
#include "../../protocols/Protocols.h"
#define RESULTTYPE DATATYPE

template<typename Pr, typename S>
void AND_BENCH_1 (Pr P,/*outputs*/ DATATYPE* result)
{

/* BufferHelper buffer_helper = BufferHelper(); */

// allocate memory for shares

auto gates_a = P.alloc_Share(NUM_INPUTS);
auto gates_b = P.alloc_Share(NUM_INPUTS);
auto gates_c = P.alloc_Share(NUM_INPUTS);

P.communicate(); // dummy communication round to simulate secret sharing

for (int i = 0; i < NUM_INPUTS; i++) {
#if FUNCTION_IDENTIFIER > 4
    #if FUNCTION_IDENTIFIER == 8
    P.prepare_mult(gates_a[i],gates_b[i], gates_c[i], ADD32, SUB32, MULT32);
    #else
    P.prepare_mult(gates_a[i],gates_b[i], gates_c[i]);
    #endif
#else
P.prepare_and(gates_a[i],gates_b[i], gates_c[i]);
#endif
/* buffer_helper.update_Buffers(P); */
/* if (i % 1000 == 0) { */
/* P.communicate(); */
/* } */
}

P.communicate();

for (int i = 0; i < NUM_INPUTS; i++) {
    #if FUNCTION_IDENTIFIER > 4
    #if FUNCTION_IDENTIFIER == 8
    P.complete_mult(gates_c[i],ADD32,SUB32);
    #else
    P.complete_mult(gates_c[i]);
    #endif
    #else
    P.complete_and(gates_c[i]);
    #endif
/* buffer_helper.update_Buffers(P); */
/* if (i % 1000 == 0) { */
/* P.communicate(); */
/* } */
}
P.communicate();

P.prepare_reveal_to_all(gates_c[0]);

P.communicate();

    #if FUNCTION_IDENTIFIER == 8
    P.complete_Reveal(gates_c[0],ADD32,SUB32);
    #else
    P.complete_Reveal(gates_c[0]);
    #endif

P.communicate();

}

  
void print_result(DATATYPE result[]) 
{
}
 
