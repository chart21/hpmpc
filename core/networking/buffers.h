#pragma once
#include "../include/pch.h"
#include "sockethelper.h"
#include <vector>
int player_id;
sender_args sending_args[num_players];
receiver_args receiving_args[num_players];

#if PRE == 1
sender_args sending_args_pre[num_players];
receiver_args receiving_args_pre[num_players];
#endif

uint64_t total_send[num_players - 1] = {0};
uint64_t total_recv[num_players - 1] = {0};
#if PRE == 1
uint64_t total_send_pre[num_players - 1] = {0};
uint64_t total_recv_pre[num_players - 1] = {0};
#endif

#if FUSE_RELU_AVG == 1 || (TRUNC_DELAYED == 1 && BIT_INJECTION_TRUNC_SIM == 1)
// Average-pool denominator folded into the fused ReLU bit injection. Also used (defaulting to 1 = a pure
// truncation) when a delayed truncation is folded into the bit injection (BIT_INJECTION_TRUNC_SIM == 1).
int curr_denom = 1;
#endif

int rounds;
int rb;
int sb;
int send_count[num_players] = {0};
int share_buffer[num_players] = {0};  // TODO: move to protocol layer
int send_count_pre[num_players] = {0};
int share_buffer_pre[num_players] = {0};  // TODO: move to protocol layer
int reveal_buffer[num_players] = {0};
int total_comm;
int* elements_per_round;
int input_length[num_players] = {0};
int reveal_length[num_players] = {0};
DATATYPE* player_input;
#if num_players == 4
#define player_multiplier 2
#else
#define player_multiplier 1
#endif
#if MAL == 1
DATATYPE* verify_buffer[num_players * player_multiplier];  // Verify buffer for each player
uint64_t verify_buffer_index[num_players * player_multiplier] = {0};

#if DATTTYPE > 32
alignas(sizeof(DATATYPE)) uint32_t hash_val[num_players * player_multiplier][8];  // Hash value for each player
#else
uint32_t hash_val[num_players * player_multiplier][8];  // Hash value for each player
#endif
uint64_t elements_to_compare[num_players * player_multiplier] = {0};
#endif
#if (PRE == 1 && HAS_POST_PROTOCOL == 1) || \
    BEAVER == 1  // Store preprocessed-output to get the correct results during post-processing

#if BEAVER == 1 && PRE == 1
DATATYPE** preprocessed_outputs_bool = nullptr;
DATATYPE** preprocessed_outputs_arithmetic = nullptr;
uint64_t* preprocessed_outputs_bool_index = nullptr;
uint64_t* preprocessed_outputs_bool_input_index = nullptr;
uint64_t* preprocessed_outputs_arithmetic_input_index = nullptr;
uint64_t* preprocessed_outputs_arithmetic_index = nullptr;
#endif
DATATYPE* preprocessed_outputs = nullptr;
uint64_t preprocessed_outputs_input_index = 0;
uint64_t preprocessed_outputs_index = 0;
uint64_t total_preprocessed_outputs = 0;
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
// MODELWEIGHTS_KNOWN: in PRE, P1 freely picks its conv/FC triple share [lxly]_2 = r1 (a fresh PSELF random),
// derives its output mask l_P1 = TRUNC(-r1) from it, and pushes r1 here so the triple generation forces P1's
// share to r1 (mwk_fix_p1_share in core/generate_beaver_tiples.hpp). This keeps l_P1 and
// [lxly]_2 consistent for the reveal.
std::vector<DATATYPE> g_mwk_p1_masks;
// The GEMM (programs/functions/GEMM.hpp) is TILE_SIZE-tiled, so mask_and_send is called in TILE order, not in
// linear output-index order. Online reads the triple via retrieve_output_share_arithmetic(0, index), which is
// order-independent, but the ConvTriple buffer c[] is laid out in LINEAR output order. P1 therefore records the
// output index of each pushed r1 so the triple generation can SCATTER r1 into c[index].
std::vector<uint64_t> g_mwk_p1_indices;
#define G_MWK_LINEAR_SENTINEL ((uint64_t) -1)  // non-interleaved GEMM path: consume c[] linearly
uint64_t g_mwk_p1_masks_consume = 0;
// FC layers push into their OWN vectors: the triple generation processes ALL conv layers first and
// ALL FC layers second, so a shared vector breaks whenever conv and FC layers interleave in program
// order (the consume pointer would hand FC masks to a later conv layer and vice versa).
std::vector<DATATYPE> g_mwk_p1_fc_masks;
std::vector<uint64_t> g_mwk_p1_fc_indices;
uint64_t g_mwk_p1_fc_masks_consume = 0;
#endif
uint64_t send_in_last_round[num_players - 1] = {0};
#endif
// CUT_FRACTIONAL_BITS_OPT (see docs): under TRUNC_DELAYED == 0, this wire's true (reconstructed)
// value is provably bounded within BITLENGTH-FRACTIONAL signed bits, so the MSB adder's top
// FRACTIONAL slices are redundant. Set by RELU around its get_msb_range call. Applies regardless
// of MODELWEIGHTS_KNOWN_DURING_PREPROCESSING - the bound comes from the truncation invariant, not
// from any mask-construction trick.
bool g_cut_frac_active = false;
// Whether the values entering the current MSB extraction carry the mask bake (A2B_CONV_BAKE's committed
// mask, or RESHARE_OPT_SIM's rt.a bits in P1's mask). Only a conv/FC mask/send bakes, so the network clears
// this for ReLUs fed by anything else (BatchNorm, residual sums), and max/min clears it for comparisons.
bool g_msb_input_baked = true;
// The conv/FC being evaluated feeds a baked ReLU directly (Conv2d / Linear bake_output): only then may its output
// masks be the committed A2B masks of those ReLU inputs (A2B_CONV_BAKE). Any other conv/FC draws fresh masks: the
// committed slots are addressed by the NEXT A2B's base, so an identity-branch conv, a residual partner or the stem
// conv before a pooling layer would reuse slots of other values (their differences, or through the rebase their
// masks, would become public).
bool g_conv_bake = true;
// A2B_BAKE_MASK_PASS: the preprocessing pass's mask-only forward is running (ReLUs record their input masks and
// output the committed bit-injection masks, protocols/beaver_triples.hpp)
bool g_mask_pass = false;
// A2B_CONV_BAKE, residual sums: the ReLU input is the sum of two layer outputs; the conv/FC computed last (the partner)
// draws lz - (the other addend's mask), which ResNet's forward publishes here (NCHW, like the output), so that the
// sum carries lz. g_msb_input_residual: the running ReLU's input is such a sum.
const DATATYPE* g_bake_res_l = nullptr;
bool g_msb_input_residual = false;
// P1 with weights known in preprocessing and SecureML truncation: its conv/FC masks lie in the truncation's image, so
// the partner cannot draw lz_1 - (a free mask). Instead the other addend's P1 mask m_b is committed too (a conv/FC's: a
// PRF value in the image; a ReLU's: its committed output masks) and P1's committed mask of residual sum k becomes
// lz_1 = m_a + m_b (init_a2b_bake); the partner then draws lz_1 - m_b = m_a, in the image. Sums numbered in network
// order: the network marks the other addend's producer (g_res_producer_k while a conv/FC runs, g_relu_identity_k while
// a ReLU runs), the sum's ReLU (g_residual_k) and the partner (g_bake_res_k); the INIT pass records the slots.
int g_residual_k = -1, g_res_producer_k = -1, g_relu_identity_k = -1, g_bake_res_k = -1;
struct ResidualSum
{
    uint64_t slot_base = 0, slots = 0;  // the sum's A2B slots (INIT pass)
    int producer = 0;                    // the other addend's producer: 0 another layer or the input, 1 conv/FC, 2 ReLU
    uint64_t relu_base = 0;              // ReLU producer: its first committed output slot (g_relu_out, INIT pass)
};
std::vector<ResidualSum> g_residual_sums;
inline ResidualSum& residual_sum(int k)
{
    if ((size_t) k >= g_residual_sums.size())
        g_residual_sums.resize(k + 1);
    return g_residual_sums[k];
}
// Set by the conv/FC layers around every GEMM regardless of protocol, hence declared outside the
// preprocessing guard above.
// RESHARE_OPT / A2B_CONV_BAKE: the conv layer runs ONE GEMM per batch element, so the mask index passed to
// the indexed mask_and_send variants is layer-local per element (0..N-1), while the ReLU's MSB adders
// consume the layer's reshare/bake material globally across the batch. The layer sets this to
// (element * N) around each per-element GEMM so the bake sees the batch-global output index; FC runs a
// single GEMM with a global index, so it stays 0.
uint64_t g_bake_batch_offset = 0;
// Effective bias-mask shares published by the conv/FC layer; the bakes pre-compensate them
// (protocols/beaver_triples.hpp).
const DATATYPE* g_bake_bias_l = nullptr;
uint64_t g_bake_bias_len = 0;
uint64_t num_generated[num_players * player_multiplier] = {0};

// ADDITIONAL_RELU_THREADS (programs/functions/stream_parallel.hpp): while a worker runs elements of a
// circuit level, its sends, receives, preprocessed outputs and own randomness come from these cursors
// into the streams, at the positions the serial run would have used.
// the preprocessing streams read by index (protocols/beaver_triples.hpp); a parallel circuit level gives every
// worker its own copy of the index, at the position of its first element
enum StreamIndex
{
    IDX_BOOL,
    IDX_ARITH,
    IDX_BEAVER3,
    IDX_BEAVER4,
    IDX_RANDOM_MULT,
    IDX_ARITH_AB2,
    IDX_BOOL_AB2,
    IDX_A2B_S1_PENDING,  // g_a2b_s1_pending: P0's A2B group count (PPA4 RESHARE_OPT_SIM bake)
    IDX_COUNT
};
struct StreamCursor
{
    DATATYPE* send;
    const DATATYPE* recv;
    const DATATYPE* pre;
    const DATATYPE* pre_bool;
    const DATATYPE* pre_arith;
    const DATATYPE* rnd;
    uint64_t idx[IDX_COUNT];  // per-thread indices of the index-addressed preprocessing streams (enum StreamIndex)
};
inline thread_local StreamCursor* tl_stream = nullptr;
// The same for the preprocessing pass (PHASE_PRE): the append-only streams its elements write. A worker of a
// parallel preprocessing level writes through these; reads (randomness, retrieved triples, stored outputs) go
// through tl_stream as online. Each pointer is the element's next slot in the stream, null if the level does not use it.
struct PreCursor
{
    DATATYPE* send;                     // pre_send_to_live(PNEXT)
    uint8_t* type[2];                   // triple_type[r]
    DATATYPE *ab_bool_a, *ab_bool_b;    // storeBooleanABTriple
    DATATYPE *ab_arith_a, *ab_arith_b;  // storeArithmeticABTriple
    DATATYPE *ab2_bool, *ab2_arith;     // storeBooleanAB2Triple / storeArithmeticAB2Triple (this party's array)
    DATATYPE* bool_add;                 // boolean_addition_triple_{a,b}
    DATATYPE *mux_arith, *mux_bool;     // multiplexer_triple_{a,b}
    DATATYPE *cot_arith, *cot_bool;     // cot_triple_a (arithmetic / boolean index)
    DATATYPE* out;                      // store_output_share
    DATATYPE *out_bool[2], *out_arith[2];  // store_output_share_{bool,arithmetic}(index)
};
inline thread_local PreCursor* tl_pre = nullptr;
// the index a retrieval uses: the worker's cursor inside a parallel level, the global one otherwise
inline uint64_t& stream_index(int k, uint64_t& global) { return tl_stream ? tl_stream->idx[k] : global; }
uint64_t rnd_calls_self = 0;  // getRandomVal(PSELF) calls outside the cursors
[[noreturn]] inline void stream_cursor_misuse(const char* what)
{
    fprintf(stderr, "ADDITIONAL_RELU_THREADS: %s inside a parallel circuit level\n", what);
    abort();
}

int use_srng_for_inputs = 1;

// Set by the conv/FC layer to its is_first flag: 1 only for the network's first layer, whose input is the raw
// data-owner share (non-owner mask = 0). With PUBLIC_WEIGHTS, that layer's truncation routes to the *_a_known
// variant (owner truncates in the clear) instead of the SecureML local truncation, which wraps on the (0,value)
// sharing. See protocols/2-PC/aby2/aby2_online.hpp prepare_mult_public_fixed_a_known.
int g_a_known_input = 0;

int current_phase = 0;   // Keeping track of current pahse
int process_offset = 0;  // offsets the starting input for each process, base port must be multiple of 1000 to work

#if TRUNC_DELAYED == 1
bool delayed = false;  // For delayed truncation
bool isReLU = false;   // For ReLU truncation
#endif

#if TRUNC_APPROACH > 0
bool all_positive = false;  // for slack-based optiimzation
#endif

