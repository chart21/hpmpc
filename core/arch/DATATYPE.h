#pragma once
#include "../../config.h"
#ifndef RUNTIME
#define RUNTIME
#endif
#ifndef ORTHO
#define ORTHO
#endif
#define US

#if DATTYPE == 1
#include "BOOL.h"
#elif DATTYPE == 8
#include "STD.h"
#elif DATTYPE == 16
#include "STD.h"
#elif DATTYPE == 32
#include "STD.h"
#elif DATTYPE == 64
#include "STD.h"
#elif DATTYPE == 128
#include "SSE.h"
#elif DATTYPE == 256
#include "AVX.h"
#elif DATTYPE == 512
#include "AVX512.h"
#else
printf("Datatype not supported \n");
exit(1);
#endif

#if FUNCTION_IDENTIFIER != 2 && FUNCTION_IDENTIFIER != 412 && FUNCTION_IDENTIFIER != 413 && FUNCTION_IDENTIFIER != 414
// workaround to benchmark some functions easier

void orthogonalize_arithmetic(UINT_TYPE* in, DATATYPE* out)
{
    orthogonalize_arithmetic(in, out, BITLENGTH);
}

void unorthogonalize_arithmetic(DATATYPE* in, UINT_TYPE* out)
{
    unorthogonalize_arithmetic(in, out, BITLENGTH);
}

void orthogonalize_arithmetic_full(UINT_TYPE* in, DATATYPE* out)
{
    orthogonalize_arithmetic(in, out, DATTYPE);
}

void unorthogonalize_arithmetic_full(DATATYPE* in, UINT_TYPE* out)
{
    unorthogonalize_arithmetic(in, out, DATTYPE);
}

#endif

#if DATTYPE > 64
namespace std
{

template <>
struct bit_xor<DATATYPE>
{
    inline DATATYPE operator()(const DATATYPE a, const DATATYPE b) const { return FUNC_XOR(a, b); }
};

template <>
struct bit_and<DATATYPE>
{
    inline DATATYPE operator()(const DATATYPE a, const DATATYPE b) const { return FUNC_AND(a, b); }
};

#if BITLENGTH == 64
template <>
struct plus<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_ADD64(a, b); }
};

template <>
struct minus<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_SUB64(a, b); }
};

template <>
struct multiplies<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_MUL64(a, b); }
};

#elif BITLENGTH == 32

template <>
struct plus<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_ADD32(a, b); }
};

template <>
struct minus<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_SUB32(a, b); }
};

template <>
struct multiplies<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_MUL32(a, b); }
};

#elif BITLENGTH == 16
template <>
struct plus<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_ADD16(a, b); }
};

template <>
struct minus<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_SUB16(a, b); }
};

template <>
struct multiplies<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_MUL16(a, b); }
};

#elif BITLENGTH == 8
template <>
struct plus<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_ADD8(a, b); }
};

template <>
struct minus<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_SUB8(a, b); }
};

template <>
struct multiplies<DATATYPE>
{
    inline DATATYPE operator()(DATATYPE a, DATATYPE b) { return FUNC_MUL8(a, b); }
};

#endif
}  // namespace std
#endif

#if BITLENGTH == 8
/* #define OP_TRUNC SHIFT_RIGHT8<FRACTIONAL> */
#define OP_SHIFT_LEFT SHIFT_LEFT8
#define OP_SHIFT_RIGHT SHIFT_RIGHT8
#define OP_SHIFT_LOG_RIGHT SHIFT_LOG_RIGHT8
#define OP_SHIFT_LOG_RIGHTF SHIFT_LOG_RIGHT8F
#elif BITLENGTH == 16
#define OP_SHIFT_LEFT SHIFT_LEFT16
#define OP_SHIFT_RIGHT SHIFT_RIGHT16
#define OP_SHIFT_LOG_RIGHT SHIFT_LOG_RIGHT16
#define OP_SHIFT_LOG_RIGHTF SHIFT_LOG_RIGHT16F
#elif BITLENGTH == 32
/* #define OP_TRUNC SHIFT_RIGHT32<FRACTIONAL> */
#define OP_SHIFT_LEFT SHIFT_LEFT32
#define OP_SHIFT_RIGHT SHIFT_RIGHT32
#define OP_SHIFT_LOG_RIGHT SHIFT_LOG_RIGHT32
#define OP_SHIFT_LOG_RIGHTF SHIFT_LOG_RIGHT32F
#elif BITLENGTH == 64
#define OP_SHIFT_LEFT SHIFT_LEFT64
#define OP_SHIFT_RIGHT SHIFT_RIGHT64
#define OP_SHIFT_LOG_RIGHT SHIFT_LOG_RIGHT64
#define OP_SHIFT_LOG_RIGHTF SHIFT_LOG_RIGHT64F
#endif
#define OP_ADD std::plus<DATATYPE>()
#define OP_SUB std::minus<DATATYPE>()
#define OP_MULT std::multiplies<DATATYPE>()
#define OP_XOR std::bit_xor<DATATYPE>()
#define OP_AND std::bit_and<DATATYPE>()

/* #define OP_TRUNC OP_SHIFT_RIGHT<FRACTIONAL> */
/* #define OP_TRUNC2 OP_SHIFT_RIGHT<1> */
/* #define OP_TRUNCF OP_SHIFT_RIGHT */
#if SKIP_PRE == 0
#define OP_TRUNC OP_SHIFT_LOG_RIGHT<FRACTIONAL>
#else
#define OP_TRUNC OP_SHIFT_RIGHT<FRACTIONAL>
#endif
#define OP_TRUNC2 OP_SHIFT_LOG_RIGHT<1>
#define OP_TRUNCF OP_SHIFT_LOG_RIGHTF

DATATYPE TRUNC2(DATATYPE x)
{
    // Create a mask with lower k bits set to 1

    x = OP_TRUNC(x);
    UINT_TYPE maskValue = (UINT_TYPE(1) << (BITLENGTH - FRACTIONAL)) - 1;
    DATATYPE mask = PROMOTE(maskValue);  // Set all elements to maskValue
    // Apply the mask using bitwise AND
    return FUNC_AND(x, mask);
}

DATATYPE TRUNC3(DATATYPE x)
{
    // Create a mask with lower k bits set to 1

    x = OP_SUB(SET_ALL_ZERO(), OP_TRUNC(x));
    UINT_TYPE maskValue = (UINT_TYPE(1) << (BITLENGTH - FRACTIONAL)) - 1;
    DATATYPE mask = PROMOTE(maskValue);  // Set all elements to maskValue
    // Apply the mask using bitwise AND
    return OP_SUB(SET_ALL_ZERO(), FUNC_AND(x, mask));
}

#define FUNC_TRUNC OP_TRUNC

#if DATTYPE == 1
#if COMPRESS == 1
#define BOOL_COMPRESS
#define NEW(var) new (std::align_val_t(sizeof(uint64_t))) var;  // align variables for packing/unpacking
#else
#define NEW(var) new var;
#endif
#endif
