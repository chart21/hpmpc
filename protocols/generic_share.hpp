#pragma once
#include "../arch/DATATYPE.h"
#include "../networking/sockethelper.h"
#include "../networking/buffers.h"
#include "../utils/randomizer.h"
#if INIT == 1
    #include "init_protocol_base.hpp"
#endif
#if LIVE == 1
    #include "live_protocol_base.hpp"
#endif
#define INIT_SHARE XOR_Share
/* #define XOR_Share DATATYPE[BITLENGTH][2] */

