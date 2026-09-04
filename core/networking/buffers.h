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
// share to r1 (mwk_fix_p1_share / MWK_PRESCRIBED_HE in core/generate_beaver_tiples.hpp). This keeps l_P1 and
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

