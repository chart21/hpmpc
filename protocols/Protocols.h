#pragma once
#include "../config.h"
#include "../core/arch/DATATYPE.h"
#include "../core/networking/sockethelper.h"
#include "../core/utils/randomizer.h"

#if INIT == 1
#include "init_protocol_base.hpp"
#endif
#if LIVE == 1
#include "live_protocol_base.hpp"
#endif
#define sharemind 1
#define rep3 2
#define astra 3
#define aby2_dummy 4
#define trio 5
#define ttp3 6
#define ttp4 7
#define Tetrad 8
#define FantasticFour 9
#define Quad 10
#define Quad_het 11
#define Quad_OffOn 12
#define Simulate 13
#if PROTOCOL == rep3
#define PROTOCOL_LIVE Replicated_Share
#define PROTOCOL_INIT Replicated_init
#if INIT == 1
#include "3-PC/replicated/replicated_init_template.hpp"
#endif
#if LIVE == 1
#include "3-PC/replicated/replicated_template.hpp"
#endif
#elif PROTOCOL == sharemind
#define PROTOCOL_LIVE Sharemind_Share
#define PROTOCOL_INIT Sharemind_init
#if INIT == 1
#include "3-PC/sharemind/sharemind_init_template.hpp"
#endif
#if LIVE == 1
#include "3-PC/sharemind/sharemind_template.hpp"
#endif
#elif PROTOCOL == astra
#if PRE == 1
#if PARTY == 0
#define PROTOCOL_PRE ASTRA0_Share
#include "3-PC/astra/astra-P_0_template.hpp"
#endif
#if PARTY == 1
#define PROTOCOL_PRE -1
#endif
#if PARTY == 2
#define PROTOCOL_PRE -1
#endif
#endif
#if INIT == 1
#if PARTY == 0
#define PROTOCOL_INIT OECL0_init
#include "3-PC/ours/oecl-P_0_init_template.hpp"
#endif
#if PARTY == 1
#define PROTOCOL_INIT OECL1_init
#include "3-PC/ours/oecl-P_1_init_template.hpp"
#endif
#if PARTY == 2
#define PROTOCOL_INIT OECL2_init
#include "3-PC/ours/oecl-P_2_init_template.hpp"
#endif
#endif
#if LIVE == 1
#if PARTY == 0
#if PRE == 1
#define PROTOCOL_LIVE OECL0_POST_Share
#include "3-PC/ours/oecl-P_0-post_template.hpp"
#else
#define PROTOCOL_LIVE ASTRA0_Share
#include "3-PC/astra/astra-P_0_template.hpp"
#endif
#endif
#if PARTY == 1
#define PROTOCOL_LIVE ASTRA1_Share
#include "3-PC/astra/astra-P_1_template.hpp"
#endif
#if PARTY == 2
#define PROTOCOL_LIVE ASTRA2_Share
#include "3-PC/astra/astra-P_2_template.hpp"
#endif
#endif
#elif PROTOCOL == aby2_dummy
#if PRE == 1
#define PROTOCOL_PRE ABY2_Share
#include "2-PC/aby2_dummy/aby2.hpp"
#endif
#if INIT == 1
#define PROTOCOL_INIT ABY2_init
#include "2-PC/aby2_dummy/aby2_init.hpp"
#endif
#if LIVE == 1
#if PRE == 1
#define PROTOCOL_LIVE ABY2_POST_Share
/* #define HAS_POST_PROTOCOL 1 */
#include "2-PC/aby2_dummy/aby2_post.hpp"
#else
#define PROTOCOL_LIVE ABY2_Share
#include "2-PC/aby2_dummy/aby2.hpp"
#endif
#endif
#elif PROTOCOL == trio
#if PRE == 1
#if PARTY == 0
#define PROTOCOL_PRE OECL0_Share
#include "3-PC/ours/oecl-P_0_template.hpp"
#endif
#if PARTY == 1
#define PROTOCOL_PRE -1
#endif
#if PARTY == 2
#define PROTOCOL_PRE -1
#endif
#endif
#if INIT == 1
#if PARTY == 0
#define PROTOCOL_INIT OECL0_init
#include "3-PC/ours/oecl-P_0_init_template.hpp"
#endif
#if PARTY == 1
#define PROTOCOL_INIT OECL1_init
#include "3-PC/ours/oecl-P_1_init_template.hpp"
#endif
#if PARTY == 2
#define PROTOCOL_INIT OECL2_init
#include "3-PC/ours/oecl-P_2_init_template.hpp"
#endif
#endif
#if LIVE == 1
#if PARTY == 0
#if PRE == 1
#define PROTOCOL_LIVE OECL0_POST_Share
/* #define HAS_POST_PROTOCOL 1 */
#include "3-PC/ours/oecl-P_0-post_template.hpp"
#else
#define PROTOCOL_LIVE OECL0_Share
#include "3-PC/ours/oecl-P_0_template.hpp"
#endif
#endif
#if PARTY == 1
#define PROTOCOL_LIVE OECL1_Share
#include "3-PC/ours/oecl-P_1_template.hpp"
#endif
#if PARTY == 2
#define PROTOCOL_LIVE OECL2_Share
#include "3-PC/ours/oecl-P_2_template.hpp"
#endif
#endif
#elif PROTOCOL == ttp3 || PROTOCOL == ttp4 || PROTOCOL == Simulate
#define PROTOCOL_LIVE TTP_Share
#define PROTOCOL_INIT TTP_init
#define TTP_PROTOCOL 1
#if INIT == 1
#include "TTP/ttp_init_template.hpp"
#endif
#if LIVE == 1
#include "TTP/ttp_template.hpp"
#endif
#elif PROTOCOL == Quad || PROTOCOL == Quad_het || PROTOCOL == Quad_OffOn
#if PRE == 1
#if PARTY == 0
#define PROTOCOL_PRE -1
#endif
#if PARTY == 1
#define PROTOCOL_PRE -1
#endif
#if PARTY == 2
#define PROTOCOL_PRE -1
#endif
#if PARTY == 3
#define PROTOCOL_PRE OEC_MAL3_Share
#include "4-PC/ours/oec-mal-P_3_template.hpp"
#endif
#endif
#if INIT == 1
#if PARTY == 0
#define PROTOCOL_INIT OEC_MAL0_init
#include "4-PC/ours/oec-mal-P_0_init_template.hpp"
#endif
#if PARTY == 1
#define PROTOCOL_INIT OEC_MAL1_init
#include "4-PC/ours/oec-mal-P_1_init_template.hpp"
#endif
#if PARTY == 2
#define PROTOCOL_INIT OEC_MAL2_init
#include "4-PC/ours/oec-mal-P_2_init_template.hpp"
#endif
#if PARTY == 3
#define PROTOCOL_INIT OEC_MAL3_init
#include "4-PC/ours/oec-mal-P_3_init_template.hpp"
#endif
#endif
#if LIVE == 1
#if PARTY == 0
/* #if PRE == 1 */
/*     #define PROTOCOL_LIVE OEC_MAL0_POST */
/*     #include "oec-mal-P_0-post_template.hpp" */
/* #else */
#define PROTOCOL_LIVE OEC_MAL0_Share
#include "4-PC/ours/oec-mal-P_0_template.hpp"
/* #endif */
#endif
#if PARTY == 1
#define PROTOCOL_LIVE OEC_MAL1_Share
#include "4-PC/ours/oec-mal-P_1_template.hpp"
#endif
#if PARTY == 2
#define PROTOCOL_LIVE OEC_MAL2_Share
#include "4-PC/ours/oec-mal-P_2_template.hpp"
#endif
#if PARTY == 3
#if PRE == 1
#define PROTOCOL_LIVE OECL_MAL3_POST_Share
/* #define HAS_POST_PROTOCOL 1 */
#include "4-PC/ours/oec-mal-P_3-post_template.hpp"
#else
#define PROTOCOL_LIVE OEC_MAL3_Share
#include "4-PC/ours/oec-mal-P_3_template.hpp"
#endif
#endif
#endif
#elif PROTOCOL == Tetrad
#if PRE == 1
#if PARTY == 0
#define PROTOCOL_PRE -1
#endif
#if PARTY == 1
#define PROTOCOL_PRE -1
#endif
#if PARTY == 2
#define PROTOCOL_PRE -1
#endif
#if PARTY == 3
#define PROTOCOL_PRE Tetrad3_Share
#include "4-PC/tetrad/Tetrad-P_3_template.hpp"
#endif
#endif
#if INIT == 1
#if PARTY == 0
#define PROTOCOL_INIT OEC_MAL0_init
#include "4-PC/ours/oec-mal-P_0_init_template.hpp"
#endif
#if PARTY == 1
#define PROTOCOL_INIT OEC_MAL1_init
#include "4-PC/ours/oec-mal-P_1_init_template.hpp"
#endif
#if PARTY == 2
#define PROTOCOL_INIT OEC_MAL2_init
#include "4-PC/ours/oec-mal-P_2_init_template.hpp"
#endif
#if PARTY == 3
#define PROTOCOL_INIT OEC_MAL3_init
#include "4-PC/ours/oec-mal-P_3_init_template.hpp"
#endif
#endif
#if LIVE == 1
#if PARTY == 0
/* #if PRE == 1 */
/*     #define PROTOCOL_LIVE OEC_MAL0_POST */
/*     #include "oec-mal-P_0-post_template.hpp" */
/* #else */
#define PROTOCOL_LIVE Tetrad0_Share
#include "4-PC/tetrad/Tetrad-P_0_template.hpp"
/* #endif */
#endif
#if PARTY == 1
#define PROTOCOL_LIVE Tetrad1_Share
#include "4-PC/tetrad/Tetrad-P_1_template.hpp"
#endif
#if PARTY == 2
#define PROTOCOL_LIVE Tetrad2_Share
#include "4-PC/tetrad/Tetrad-P_2_template.hpp"
#endif
#if PARTY == 3
#if PRE == 1
#define PROTOCOL_LIVE OECL_MAL3_POST_Share
#include "4-PC/ours/oec-mal-P_3-post_template.hpp"
#else
#define PROTOCOL_LIVE Tetrad3_Share
#include "4-PC/tetrad/Tetrad-P_3_template.hpp"
#endif
#endif
#endif
#elif PROTOCOL == FantasticFour
#if INIT == 1
#define PROTOCOL_INIT Fantastic_Four_init
#include "4-PC/fantastic/Fantastic_init_template.hpp"
#endif
#if LIVE == 1
#define PROTOCOL_LIVE Fantastic_Four_Share
#include "4-PC/fantastic/Fantastic_Four_template.hpp"
#endif

#endif

