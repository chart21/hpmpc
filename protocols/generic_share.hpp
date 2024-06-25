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
#if USE_CUDA_GEMM == 1 || USE_CUDA_GEMM == 3
    #include "../core/cuda/gemm_cutlass_int.h"
#elif USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
    #include "../core/cuda/conv_cutlass_int.h"
    /* #include "../cuda/test_gemm.hpp" */
#endif

