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
    P.prepare_mult(gates_a[i],gates_b[i], gates_c[i], OP_ADD, OP_SUB, OP_MULT);
}

P.communicate();

for (int i = 0; i < NUM_INPUTS; i++) {
    P.complete_mult(gates_c[i],OP_ADD, OP_SUB);
}
P.communicate();

P.prepare_reveal_to_all(gates_c[0]);

P.communicate();

    P.complete_Reveal(gates_c[0],OP_ADD,OP_SUB);

P.communicate();

}

  
void print_result(DATATYPE result[]) 
{
}
 
