#pragma once
#include "../core/arch/DATATYPE.h"
#include "../core/networking/sockethelper.h"
#include "../core/networking/buffers.h"
#include "../core/utils/randomizer.h"
#if INIT == 1
    #include "init_protocol_base.hpp"
#endif
#if LIVE == 1
    #include "live_protocol_base.hpp"
#endif
#if USE_CUDA_GEMM > 0
    #include "../core/cuda/gemm_cutlass_int.h"
#endif
#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
    #include "../core/cuda/conv_cutlass_int.h"
#endif

#if num_players == 2
#define PNEXT 0
#define PPREV 0
#define PSELF 1
#if PARTY == 0
#define P_0 1
#define P_1 0
#elif PARTY == 1
#define P_0 0
#define P_1 1
#endif
#elif num_players == 3
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
#if PROTOCOL == 4
#define HAS_POST_PROTOCOL 1
#elif (PROTOCOL == 3 || PROTOCOL == 5) && PARTY == 0
#define HAS_POST_PROTOCOL 1
#elif (PROTOCOL == 8 || PROTOCOL == 11 || PROTOCOL == 12) && PARTY == 3
#define HAS_POST_PROTOCOL 1
#endif

