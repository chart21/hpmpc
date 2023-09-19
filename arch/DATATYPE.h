#pragma once
#include "../config.h"
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

#if DATTYPE != BITLENGTH
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
#endif

#if FUNCTION_IDENTIFIER < 5
#define OP_ADD FUNC_XOR
#define OP_SUB FUNC_XOR
#define OP_MULT FUNC_AND
#elif FUNCTION_IDENTIFIER == 5 || FUNCTION_IDENTIFIER == 7 || FUNCTION_IDENTIFIER == 9
#define OP_ADD FUNC_ADD32
#define OP_SUB FUNC_SUB32
#define OP_MULT FUNC_MUL32
#elif FUNCTION_IDENTIFIER == 6 || FUNCTION_IDENTIFIER == 10
#define OP_ADD FUNC_ADD64
#define OP_SUB FUNC_SUB64
#define OP_MULT FUNC_MUL64
#else
#if BITLENGTH == 32
#define OP_ADD FUNC_ADD32
#define OP_SUB FUNC_SUB32
#define OP_MULT FUNC_MUL32
#define OP_TRUNC SHIFT_RIGHT32
#elif BITLENGTH == 64
#define OP_ADD FUNC_ADD64
#define OP_SUB FUNC_SUB64
#define OP_MULT FUNC_MUL64
#define OP_TRUNC SHIFT_RIGHT64
#endif
#endif

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
#if (PROTOCOL == 4 || PROTOCOL == 5) && PARTY == 0
#define HAS_POST_PROTOCOL 1
#elif PROTOCOL == 12 && PARTY == 3
#define HAS_POST_PROTOCOL 1
#endif

