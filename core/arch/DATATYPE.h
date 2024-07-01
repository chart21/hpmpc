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

#if FUNCTION_IDENTIFIER !=2 && FUNCTION_IDENTIFIER != 412 && FUNCTION_IDENTIFIER != 413 && FUNCTION_IDENTIFIER != 414
                            //workaround to benchmark some functions easier

void orthogonalize_arithmetic(UINT_TYPE *in, DATATYPE *out)
{
    orthogonalize_arithmetic(in, out, BITLENGTH);
}

void unorthogonalize_arithmetic(DATATYPE *in, UINT_TYPE *out)
{
    unorthogonalize_arithmetic(in, out, BITLENGTH);
}

void orthogonalize_arithmetic_full(UINT_TYPE *in, DATATYPE *out)
{
    orthogonalize_arithmetic(in, out, DATTYPE);
}

void unorthogonalize_arithmetic_full(DATATYPE *in, UINT_TYPE *out)
{
    unorthogonalize_arithmetic(in, out, DATTYPE);
}

#else //Workarounds for easier benchmarking of some functions that don't need these 
template<typename a, typename b>
void orthogonalize_arithmetic(a *in, b *out)
{
}
template<typename a, typename b>
void unorthogonalize_arithmetic(a *in, b *out)
{
}
template<typename a, typename b>
void orthogonalize_boolean(a *in, b *out)
{
}
template<typename a, typename b>
void unorthogonalize_boolean(a *in, b *out)
{
}
#endif

#if FUNCTION_IDENTIFIER == 1 || FUNCTION_IDENTIFIER == 4
#define OP_ADD FUNC_XOR
#define OP_SUB FUNC_XOR
#define OP_MULT FUNC_AND
#elif FUNCTION_IDENTIFIER == 2 || FUNCTION_IDENTIFIER == 5
#define OP_ADD FUNC_ADD32
#define OP_SUB FUNC_SUB32
#define OP_MULT FUNC_MUL32
#define OP_TRUNC SHIFT_RIGHT32<FRACTIONAL>
#elif FUNCTION_IDENTIFIER == 3 || FUNCTION_IDENTIFIER == 6
#define OP_ADD FUNC_ADD64
#define OP_SUB FUNC_SUB64
#define OP_MULT FUNC_MUL64
#define OP_TRUNC SHIFT_RIGHT64<FRACTIONAL>
#endif

#if FUNCTION_IDENTIFIER == 7
#define OP_ADD FUNC_XOR
#define OP_SUB FUNC_XOR
#define OP_MULT FUNC_AND
#elif FUNCTION_IDENTIFIER == 8
#define OP_ADD FUNC_ADD32
#define OP_SUB FUNC_SUB32
#define OP_MULT FUNC_MUL32
#define OP_TRUNC SHIFT_RIGHT32<FRACTIONAL>
#elif FUNCTION_IDENTIFIER == 9
#define OP_ADD FUNC_ADD64
#define OP_SUB FUNC_SUB64
#define OP_MULT FUNC_MUL64
#define OP_TRUNC SHIFT_RIGHT64<FRACTIONAL>
#endif


#if BITLENGTH == 8
#define OP_ADD FUNC_ADD8
#define OP_SUB FUNC_SUB8
#define OP_MULT FUNC_MUL8
#define OP_TRUNC SHIFT_RIGHT8<FRACTIONAL>
#define OP_SHIFT_LEFT SHIFT_LEFT8
#define OP_SHIFT_RIGHT SHIFT_RIGHT8
#define OP_SHIFT_LOG_RIGHT SHIFT_LOG_RIGHT8
#define OP_TRUNC2 SHIFT_RIGHT8<1>
/* #define OP_TRUNC4 SHIFT_RIGHT8<3> */
/* #define OP_TRUNC8 SHIFT_RIGHT8<4> */
/* #define OP_TRUNC16 SHIFT_RIGHT8<5> */
#elif BITLENGTH == 16
#define OP_ADD FUNC_ADD16
#define OP_SUB FUNC_SUB16
#define OP_MULT FUNC_MUL16
#define OP_TRUNC SHIFT_RIGHT16<FRACTIONAL>
#define OP_SHIFT_LEFT SHIFT_LEFT16
#define OP_SHIFT_RIGHT SHIFT_RIGHT16
#define OP_SHIFT_LOG_RIGHT SHIFT_LOG_RIGHT16
#define OP_TRUNC2 SHIFT_RIGHT16<1>
/* #define OP_TRUNC4 SHIFT_RIGHT16<3> */
/* #define OP_TRUNC8 SHIFT_RIGHT16<4> */
/* #define OP_TRUNC16 SHIFT_RIGHT16<5> */
#elif BITLENGTH == 32
#define OP_ADD FUNC_ADD32
#define OP_SUB FUNC_SUB32
#define OP_MULT FUNC_MUL32
#define OP_TRUNC SHIFT_RIGHT32<FRACTIONAL>
#define OP_SHIFT_LEFT SHIFT_LEFT32
#define OP_SHIFT_RIGHT SHIFT_RIGHT32
#define OP_SHIFT_LOG_RIGHT SHIFT_LOG_RIGHT32
#define OP_TRUNC2 SHIFT_RIGHT32<1>
/* #define OP_TRUNC4 SHIFT_RIGHT32<3> */
/* #define OP_TRUNC8 SHIFT_RIGHT32<4> */
/* #define OP_TRUNC16 SHIFT_RIGHT32<5> */
#elif BITLENGTH == 64
#define OP_ADD FUNC_ADD64
#define OP_SUB FUNC_SUB64
#define OP_MULT FUNC_MUL64
#define OP_TRUNC SHIFT_RIGHT64<FRACTIONAL>
#define OP_SHIFT_LEFT SHIFT_LEFT64
#define OP_SHIFT_RIGHT SHIFT_RIGHT64
#define OP_SHIFT_LOG_RIGHT SHIFT_LOG_RIGHT64
#define OP_TRUNC2 SHIFT_RIGHT64<1>
/* #define OP_TRUNC4 SHIFT_RIGHT64<3> */
/* #define OP_TRUNC8 SHIFT_RIGHT64<4> */
/* #define OP_TRUNC16 SHIFT_RIGHT64<5> */
#endif

DATATYPE TRUNC2(DATATYPE x) {
    // Create a mask with lower k bits set to 1
    
    x = OP_TRUNC(x);
    UINT_TYPE maskValue = (UINT_TYPE(1) << (BITLENGTH-FRACTIONAL)) - 1; 
    DATATYPE mask = PROMOTE(maskValue); // Set all elements to maskValue
    // Apply the mask using bitwise AND
    return FUNC_AND(x, mask);
}

DATATYPE TRUNC3(DATATYPE x) {
    // Create a mask with lower k bits set to 1

    /* return OP_SUB(SET_ALL_ZERO(), TRUNC2(OP_SUB(SET_ALL_ZERO(), x))); */
    /* x = OP_SUB(SET_ALL_ZERO(),OP_TRUNC(OP_SUB(SET_ALL_ZERO(),x))); */
    x = OP_SUB(SET_ALL_ZERO(),OP_TRUNC(x));
    UINT_TYPE maskValue = (UINT_TYPE(1) << (BITLENGTH-FRACTIONAL)) - 1; 
    DATATYPE mask = PROMOTE(maskValue); // Set all elements to maskValue
    // Apply the mask using bitwise AND
    return OP_SUB(SET_ALL_ZERO(), FUNC_AND(x, mask));
}




#define FUNC_TRUNC OP_TRUNC

/* DATATYPE mod_power_of_2(DATATYPE x, int k) { */
/*     // Create a mask with lower k bits set to 1 */
    
/*     x = OP_TRUNC(x, k); */
/*     UINT_TYPE maskValue = (1 << k) - 1; */ 
/*     DATATYPE mask = PROMOTE(maskValue); // Set all elements to maskValue */
/*     // Apply the mask using bitwise AND */
/*     return FUNC_AND(x, mask); */
/* } */


#if DATTYPE == 1
    #if COMPRESS == 1
        #define BOOL_COMPRESS
        #define NEW(var) new (std::align_val_t(sizeof(uint64_t))) var; //align variables for packing/unpacking
    #else
        #define NEW(var) new var;
    #endif
#endif







#if num_players == 3
    #define PSELF 2
    #if PARTY == 0
        #define P_0 2
        #define P_1 0
        #define P_2 1
        #define PPREV 1
        #define PNEXT 0
    #elif PARTY == 1
        #define P_0 0
        #define P_1 2
        #define P_2 1
        #define PPREV 0
        #define PNEXT 1
    #elif PARTY == 2
        #define P_0 0
        #define P_1 1
        #define P_2 2
        #define PPREV 1
        #define PNEXT 0
    #endif
#elif num_players == 4
    #define PSELF 3
    #define P_0123 3
    #define P_012 4
    #define P_013 5
    #define P_023 6
    #define P_123 7
    #define P_123_2 3 // Trick for Protocols 10-12

    #if PARTY == 0
        #define P_0 3
        #define P_1 0
        #define P_2 1
        #define P_3 2
        #define PPREV 2
        #define PNEXT 0
        #define PMIDDLE 1
    #elif PARTY == 1
        #define P_0 0
        #define P_1 3
        #define P_2 1
        #define P_3 2
        #define PPREV 0
        #define PNEXT 1
        #define PMIDDLE 2
    #elif PARTY == 2
        #define P_0 0
        #define P_1 1
        #define P_2 3
        #define P_3 2
        #define PPREV 1
        #define PNEXT 2
        #define PMIDDLE 0
    #elif PARTY == 3
        #define P_0 0
        #define P_1 1
        #define P_2 2
        #define P_3 3
        #define PPREV 2
        #define PNEXT 0
        #define PMIDDLE 1
    #endif
#endif


    //temporary solution
#if (PROTOCOL == 3 || PROTOCOL == 4 || PROTOCOL == 5) && PARTY == 0
#define HAS_POST_PROTOCOL 1
#elif (PROTOCOL == 8 || PROTOCOL == 11 || PROTOCOL == 12) && PARTY == 3
#define HAS_POST_PROTOCOL 1
#endif
