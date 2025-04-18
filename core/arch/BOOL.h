/* ******************************************** *\
 *
 *
 *  DATATYPE: the base type of every value.
 *  SDATATYPE: the signed version of DATATYPE.
\* ******************************************** */

/* Including headers */
#pragma once
#include "../../config.h"
#include "../include/pch.h"

#ifndef BOOL
#define BOOL
#endif
#ifndef BITS_PER_REG
#define BITS_PER_REG 1
#endif
/* Defining 0 and 1 */
#define ZERO false
#define ONES true

#define SET_ALL_ONE() true
#define SET_ALL_ZERO() false

#define AND(a, b) ((a) && (b))
#define OR(a, b) ((a) || (b))
#define XOR(a, b) ((a) ^ (b))
#define ANDN(a, b) (!(a) && (b))
#define NOT(a) (!(a))

/* #define ADD(a,b,c) ((a) + (b)) */
/* #define SUB(a,b,c) ((a) - (b)) */

#define MUL_SIGNED(a, b, c) AND(a, b)
#define ADD_SIGNED(a, b, c) XOR(a, b)
#define SUB_SIGNED(a, b, c) XOR(a, b)

#define FUNC_AND std::bit_and<bool>()
#define FUNC_OR std::bit_or<bool>()
#define FUNC_XOR std::bit_xor<bool>()
#define FUNC_NOT std::bit_not<bool>()
#define FUNC_ADD32 std::bit_xor<bool>()
#define FUNC_SUB32 std::bit_xor<bool>()
#define FUNC_MUL32 std::bit_and<bool>()
#define FUNC_ADD64 std::bit_xor<bool>()
#define FUNC_SUB64 std::bit_xor<bool>()
#define FUNC_MUL64 std::bit_and<bool>()

#define SHIFT_RIGHT8 SET_ZERO
#define SHIFT_RIGHT16 SET_ZERO
#define SHIFT_RIGHT32 SET_ZERO
#define SHIFT_RIGHT64 SET_ZERO
#define SHIFT_LOG_RIGHT8 SET_ZERO
#define SHIFT_LOG_RIGHT16 SET_ZERO
#define SHIFT_LOG_RIGHT32 SET_ZERO
#define SHIFT_LOG_RIGHT64 SET_ZERO
#define SHIFT_LEFT8 SET_ZERO
#define SHIFT_LEFT16 SET_ZERO
#define SHIFT_LEFT32 SET_ZERO
#define SHIFT_LEFT64 SET_ZERO
#define SHIFT_RIGHT8F SET_ZERO
#define SHIFT_RIGHT16F SET_ZERO
#define SHIFT_RIGHT32F SET_ZERO
#define SHIFT_RIGHT64F SET_ZERO
#define SHIFT_LOG_RIGHT8F SET_ZERO
#define SHIFT_LOG_RIGHT16F SET_ZERO
#define SHIFT_LOG_RIGHT32F SET_ZERO
#define SHIFT_LOG_RIGHT64F SET_ZERO
#define SHIFT_LOG_RIGHT8F SET_ZERO
#define SHIFT_LOG_RIGHT16F SET_ZERO
#define SHIFT_LOG_RIGHT32F SET_ZERO
#define SHIFT_LOG_RIGHT64F SET_ZERO

template <int n>
bool SET_ZERO(bool value)
{
    return false;
}

bool SET_ZERO(bool value, int n)
{
    return false;
}

#ifndef DATATYPE
#define DATATYPE bool
#endif

#define PROMOTE(x) (SET_ALL_ONE())

#define ORTHOGONALIZE(in, out) orthogonalize(in, out)
#define UNORTHOGONALIZE(in, out) unorthogonalize(in, out)

#define ALLOC(size) malloc(size * sizeof(bool))

void orthogonalize_boolean(UINT_TYPE* num, bool* out)
{
    for (int i = 0; i < BITLENGTH; i++)
    {
        out[BITLENGTH - i - 1] = *num & 1;
        *num /= 2;
    }
}

// convert bool array to uint number
void unorthogonalize_boolean(bool* arr, UINT_TYPE* num)
{
    UINT_TYPE tmp;
    for (int i = 0; i < BITLENGTH; i++)
    {
        tmp = arr[i];
        *num |= tmp << (BITLENGTH - i - 1);
    }
}

void orthogonalize_arithmetic(UINT_TYPE* num, bool* out, int bitlength)
{
    for (int i = 0; i < bitlength; i++)
    {
        out[bitlength - i - 1] = *num & 1;
        *num /= 2;
    }
}

void unorthogonalize_arithmetic(const bool* arr, UINT_TYPE* num, int bitlength)
{
    UINT_TYPE tmp;
    for (int i = 0; i < bitlength; i++)
    {
        tmp = arr[i];
        *num |= tmp << (bitlength - i - 1);
    }
}
