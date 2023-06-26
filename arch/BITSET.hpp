
/* ******************************************** *\
 *
 *
 *  DATATYPE: the base type of every value.
 *  SDATATYPE: the signed version of DATATYPE.
\* ******************************************** */


/* Including headers */
#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
/* #include <stdbool.h> */
#include <bitset>
#ifndef BITSET
#define BITSET
#endif
#define BITSIZE 64
#ifndef BITS_PER_REG
#define BITS_PER_REG 1
#endif
/* Defining 0 and 1 */
#define ZERO false
#define ONES true

/* Defining macros */

#define AND(a,b)  ((a) && (b))
#define OR(a,b)   ((a) || (b))
#define XOR(a,b)  ((a) ^ (b))
#define ANDN(a,b) (!(a) && (b))
#define NOT(a)    (!(a))

#define ADD(a,b,c) ((a) + (b))
#define SUB(a,b,c) ((a) - (b))

#ifndef DATATYPE
#define DATATYPE std::bitset<BITSIZE> 
#endif
#define SET_ALL_ONE() set()
#define SET_ALL_ZERO() reset()

#define ORTHOGONALIZE(in,out)   orthogonalize(in,out)
#define UNORTHOGONALIZE(in,out) unorthogonalize(in,out)

#define ALLOC(size) malloc(size * sizeof(bool))
#define NEW(var) new var;


void orthogonalize(uint64_t* num, std::bitset<BITSIZE> *out) {
    std::bitset<64> tmp(*num);
    *out = tmp;
}

//convert bool array to uint64 number
void unorthogonalize(std::bitset<BITSIZE> *arr, uint64_t *num) {
*num = arr->to_ullong();
}

