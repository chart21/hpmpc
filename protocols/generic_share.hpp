#pragma once
#include "../arch/DATATYPE.h"
#include "../networking/sockethelper.h"
#include "../networking/buffers.h"
#include "../utils/randomizer.h"
#include <iostream>
#if INIT == 1
    #include "init_protocol_base.hpp"
#endif
#if LIVE == 1
    #include "live_protocol_base.hpp"
#endif
#if USE_CUDA_GEMM == 1
    #include "../cuda/gemm_cutlass_int.h"
    /* #include "../cuda/test_gemm.hpp" */
#endif

