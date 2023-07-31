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
    #include "CHAR.h"
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
        #define P0 2
        #define P1 0
        #define P2 1
        #define PPREV 1
        #define PNEXT 0
    #elif PARTY == 1
        #define P0 0
        #define P1 2
        #define P2 1
        #define PPREV 0
        #define PNEXT 1
    #elif PARTY == 2
        #define P0 0
        #define P1 1
        #define P2 2
        #define PPREV 1
        #define PNEXT 0
    #endif
#elif num_players == 4
    #define PSELF 3
    #define P0123 3
    #define P012 4
    #define P013 5
    #define P023 6
    #define P123 7
    #define P123_2 3 // Trick for Protocols 10-12

    #if PARTY == 0
        #define P0 3
        #define P1 0
        #define P2 1
        #define P3 2
        #define PPREV 2
        #define PNEXT 0
        #define PMIDDLE 1
    #elif PARTY == 1
        #define P0 0
        #define P1 3
        #define P2 1
        #define P3 2
        #define PPREV 0
        #define PNEXT 1
        #define PMIDDLE 2
    #elif PARTY == 2
        #define P0 0
        #define P1 1
        #define P2 3
        #define P3 2
        #define PPREV 1
        #define PNEXT 2
        #define PMIDDLE 0
    #elif PARTY == 3
        #define P0 0
        #define P1 1
        #define P2 2
        #define P3 3
        #define PPREV 2
        #define PNEXT 0
        #define PMIDDLE 1
    #endif
#endif

