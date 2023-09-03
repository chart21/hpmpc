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
#include "../config.h"

#ifndef STD
#define STD
#endif

#ifndef BITS_PER_REG
#define BITS_PER_REG DATTYPE
#endif
#ifndef LOG2_BITS_PER_REG
#define LOG2_BITS_PER_REG LOG2_DATATYPE
#endif

/* Defining 0 and 1 */
#define ZERO 0
#define ONES -1

#define PROMOTE(x) (x)

/* Defining macros */
#define REG_SIZE BITS_PER_REG
#define CHUNK_SIZE 64

#define AND(a,b)  ((a) & (b))
#define OR(a,b)   ((a) | (b))
#define XOR(a,b)  ((a) ^ (b))
#define ANDN(a,b) (~(a) & (b))
#define NOT(a)    (~(a))

/* #define ADD(a,b,c) ((a) + (b)) */
/* #define SUB(a,b,c) ((a) - (b)) */

#define MUL_SIGNED(a,b,c) a * b
#define ADD_SIGNED(a,b,c) a + b
#define SUB_SIGNED(a,b,c) a - b

#define FUNC_AND  std::bit_and<uint64_t>()
#define FUNC_OR   std::bit_or<uint64_t>()
#define FUNC_XOR  std::bit_xor<uint64_t>()
#define FUNC_NOT  std::bit_not<uint64_t>()
#define FUNC_ADD32 std::plus<uint64_t>()
#define FUNC_SUB32 std::minus<uint64_t>()
#define FUNC_MUL32 std::multiplies<uint64_t>()
#define FUNC_ADD64 std::plus<uint64_t>()
#define FUNC_SUB64 std::minus<uint64_t>()
#define FUNC_MUL64 std::multiplies<uint64_t>()

#define SHIFT_RIGHT32 arithmetic_right_shift_32
#define SHIFT_RIGHT64 arithmetic_right_shift_64
#define SHIFT_LEFT32(a) ((a) << FRACTIONAL)
#define SHIFT_LEFT64(a) ((a) << FRACTIONAL)

#if BITLENGTH == 64
uint64_t arithmetic_right_shift_64(uint64_t value) {
    if (value & 0x8000000000000000) {
        uint64_t mask = ~((1ULL << (64 - FRACTIONAL)) - 1);
        return value >> FRACTIONAL | mask;
    }
    return value >> FRACTIONAL;
}

#elif BITLENGTH == 32

uint32_t arithmetic_right_shift_32(uint32_t value) {
    if (value & 0x80000000) {
        uint32_t mask = ~((1U << (32 - FRACTIONAL)) - 1);
        return value >> FRACTIONAL | mask;
    }
    return value >> FRACTIONAL;
}
#endif

#define ROTATE_MASK(x) (x == 64 ? -1ULL : x == 32 ? -1 : x == 16 ? 0xFFFF : \
    ({ fprintf(stderr,"Not implemented rotate [uint%d_t]. Exiting.\n",x); \
      exit(1); 1; }))

#define L_SHIFT(a,b,c) (c == 4 ? ((a) << (b)) & 0xf : ((a) << (b)))
#define R_SHIFT(a,b,c) ((a) >> (b))
#define RA_SHIFT(a,b,c) (((SDATATYPE)(a)) >> (b))
#define L_ROTATE(a,b,c) ((a << b) | ((a&ROTATE_MASK(c)) >> (c-b)))
#define R_ROTATE(a,b,c) (((a&ROTATE_MASK(c)) >> b) | (a << (c-b)))

#define LIFT_4(x)  (x)
#define LIFT_8(x)  (x)
#define LIFT_16(x) (x)
#define LIFT_32(x) (x)
#define LIFT_64(x) (x)

#define BITMASK(x,n,c) -(((x) >> (n)) & 1)

#define PACK_8x2_to_16(a,b)  ((((uint16_t)(a)) << 8) | ((uint16_t) (b)))
#define PACK_16x2_to_32(a,b) ((((uint32_t)(a)) << 16) | ((uint32_t) (b)))
#define PACK_32x2_to_64(a,b) ((((uint64_t)(a)) << 32) | ((uint64_t) (b)))


#define refresh(x,y) *(y) = x

#ifndef DATATYPE
#if BITS_PER_REG == 4
#define DATATYPE uint8_t // TODO: use something else? do something else?
                         // (needed for Photon right now)
#define SDATATYPE int8_t
#elif DATTYPE == 8
#define DATATYPE uint8_t
#define SDATATYPE int8_t
#elif DATTYPE == 16
#define DATATYPE uint16_t
#define SDATATYPE int16_t
#elif DATTYPE == 32
#define DATATYPE uint32_t
#define SDATATYPE int32_t
#else
#define DATATYPE uint64_t
#define SDATATYPE int64_t
#endif
#endif

#define SET_ALL_ONE()  -1
#define SET_ALL_ZERO() 0

#define ORTHOGONALIZE(in,out)   orthogonalize(in,out)
#define UNORTHOGONALIZE(in,out) unorthogonalize(in,out)

#define ALLOC(size) malloc(size * sizeof(uint64_t))
/* #define NEW(var) (sizeof(var) > 0) ? new var : NULL */ 
#define NEW(var) new var 




/* Orthogonalization stuffs */
static uint64_t mask_l[6] = {
	0xaaaaaaaaaaaaaaaaUL,
	0xccccccccccccccccUL,
	0xf0f0f0f0f0f0f0f0UL,
	0xff00ff00ff00ff00UL,
	0xffff0000ffff0000UL,
	0xffffffff00000000UL
};

static uint64_t mask_r[6] = {
	0x5555555555555555UL,
	0x3333333333333333UL,
	0x0f0f0f0f0f0f0f0fUL,
	0x00ff00ff00ff00ffUL,
	0x0000ffff0000ffffUL,
	0x00000000ffffffffUL
};


void real_ortho(UINT_TYPE data[]) {
  for (int i = 0; i < LOG2_BITLENGTH; i ++) {
    int nu = (1UL << i);
    for (int j = 0; j < BITLENGTH; j += (2 * nu))
      for (int k = 0; k < nu; k ++) {
        UINT_TYPE u = data[j + k] & mask_l[i];
        UINT_TYPE v = data[j + k] & mask_r[i];
        UINT_TYPE x = data[j + nu + k] & mask_l[i];
        UINT_TYPE y = data[j + nu + k] & mask_r[i];
        data[j + k] = u | (x >> nu);
        data[j + nu + k] = (v << nu) | y;
      }
  }
}


void orthogonalize_boolean(UINT_TYPE* data, DATATYPE* out) {
  for (int i = 0; i < BITLENGTH; i++)
    out[i] = ((DATATYPE*) data)[i];
  real_ortho(out);
}

void unorthogonalize_boolean(DATATYPE *in, UINT_TYPE* data) {
  for (int i = 0; i < DATTYPE; i++)
    data[i] = ((UINT_TYPE*) in)[i];
  real_ortho(data);
}

// STD does not allow arithmetic of packed integers -> only allow DATATYPE in/out

void orthogonalize_arithmetic(DATATYPE* data, DATATYPE* out) {
  for (int i = 0; i < BITLENGTH; i++)
    out[i] = data[i];
}

    void unorthogonalize_arithmetic(DATATYPE *in, DATATYPE* data) {
  for (int i = 0; i < DATTYPE; i++)
    data[i] = in[i];
}


void orthogonalize_boolean_full(UINT_TYPE* data, DATATYPE* out) {
  for (int i = 0; i < DATTYPE; i++)
    out[i] = data[i];
  real_ortho(out);
}

void unorthogonalize_boolean_full(DATATYPE *in, UINT_TYPE* data) {
  for (int i = 0; i < DATTYPE; i++)
    data[i] = in[i];
  real_ortho(data);
}

void orthogonalize_arithmetic_full(DATATYPE* data, DATATYPE* out) {
  for (int i = 0; i < DATTYPE; i++)
    out[i] = data[i];
}

void unorthogonalize_arithmetic_full(DATATYPE *in, DATATYPE* data) {
  for (int i = 0; i < DATTYPE; i++)
    data[i] = in[i];
}

