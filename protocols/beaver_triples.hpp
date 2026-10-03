#pragma once
#include <functional>
#include <map>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "../core/generate_beaver_tiples.hpp"
#include "../core/init.hpp"
#include "../config.h"  
#include "generic_share.hpp"
#include "../core/include/pch.h"

template <typename Datatype>
struct Beaver3Tuple {
    Datatype a;
    Datatype b;
    Datatype c;
    Datatype ab;
    Datatype ac;
    Datatype bc;
    Datatype abc;
};

template <typename Datatype>
struct Beaver4Tuple {
    Datatype a;
    Datatype b;
    Datatype c;
    Datatype d;
    Datatype ab;
    Datatype ac;
    Datatype ad;
    Datatype bc;
    Datatype bd;
    Datatype cd;
    Datatype abc;
    Datatype abd;
    Datatype acd;
    Datatype bcd;
    Datatype abcd;
};

template <typename Datatype>
struct RandomMultiplication {
    Datatype a;
    Datatype b;
};

// Compile-time check: is bit index i reshared in a PPA4Way circuit of width k?
// The reshared positions correspond to AND2 "generate" gates g1 = a[i] & b[i]
// at each level of the 4-way prefix tree.
constexpr bool is_ppa4_reshared(int k, int i)
{
    if (k == 8)
    {
        return i == 1 || i == 2 || i == 5;
    }
    else if (k == 16)
    {
        return i == 1 || i == 4 || i == 7 || i == 10 || i == 13;
    }
    else if (k == 32)
    {
        return i == 1 || i == 4 || i == 7 || i == 10 || i == 13
            || i == 16 || i == 19 || i == 22 || i == 23 || i == 26 || i == 29;
    }
    // k = 64 and the narrow cut adders of 64 - FRACTIONAL bits (scripts/circuits/gen_64bit_adders.py, which checks the
    // generated circuits against this): the first slice of each level-0 group of 3
    return k > 32 && i >= 1 && i < k && i % 3 == 1;
}

// std::vector<uint64_t> arithmetic_triple_index;
// std::vector<uint64_t> boolean_triple_index;
uint64_t num_beaver_3_tuples;
uint64_t num_beaver_4_tuples;
uint64_t num_random_multiplications = 0;
uint64_t curr_beaver_3_triple_index = 0;
uint64_t curr_beaver_4_triple_index = 0;
uint64_t curr_random_multiplication_index = 0;
// RESHARE_OPT_SIM validation counters: the sim is bit-identical to SIM=0 iff P1's A2B slice l equals its rt.a
// share at every reshare_b (checked live on P1, see reshare_sim_check). This is the SENSITIVE correctness check:
// end-to-end tests can miss LSB-slice errors (an RCA carry error only shifts the sum by +-2, which almost never
// flips the MSB). Residual mismatches on padded groups (layer size not a multiple of BITLENGTH) are EXPECTED:
// the padding lanes belong to no value and cannot be baked; they are harmless (bit-sliced gates are lane-local).
uint64_t g_rb_checks = 0, g_rb_mismatch = 0;
uint64_t beaver_3_triple_index = 0;
uint64_t beaver_4_triple_index = 0;
uint64_t random_multiplication_index = 0;
Beaver3TuplesD<DATATYPE> beaver_3_tuples;
Beaver4TuplesD<DATATYPE> beaver_4_tuples;
DATATYPE* random_multiplication_a = nullptr;
DATATYPE* random_multiplication_b = nullptr;

// All reshare-baking machinery below is active only in this configuration; call sites can rely on
// bake_reshare_mask compiling to a no-op otherwise (construct_mwk_r1_baked call sites must still be
// gated because they consume an extra PRNG draw).
// (BITLENGTH 32 only: the slot maps below, reshare_rt_offset etc., are those of the 32-bit circuits)
#define RESHARE_BAKE_ACTIVE \
    (RESHARE_OPT == 1 && RESHARE_OPT_SIM == 1 && DATTYPE == BITLENGTH && (BITLENGTH == 32 || BITLENGTH == 64) && \
     (RCA_MSB == 1 || PPA_MSB == 1 || PPA4_MSB == 1))

// CUT_FRACTIONAL_BITS_OPT (docs/CUT_FRACTIONAL_BITS_OPT.md): compile-time eligibility. Under
// TRUNC_DELAYED == 0 the ReLU input is freshly truncated, so its value fits BITLENGTH-FRACTIONAL
// signed bits and the MSB adder's top FRACTIONAL slices are redundant. All nine 2PC msb circuits
// (ripple-carry / prefix / four-way prefix, each plain, reshared and a_known-to-evaluators) implement
// the cut, so eligibility depends only on that value-level precondition and the width. Whether a given
// adder instance applies it is the RUNTIME flag g_cut_frac_active: set by RELU, and by the comparison
// adders only under the SIM bake (max_min.hpp); other max/min adders run the full circuit.
#define CUT_FRAC_ELIGIBLE_PPA4 (CUT_FRAC_ELIGIBLE && PPA4_MSB == 1)

// Public-weight layers multiply locally and never bake a mask.
inline bool msb_input_baked() { return g_msb_input_baked && PUBLIC_WEIGHTS == 0; }
// The same for a residual sum's partner (A2B_BAKE_RESIDUAL): a conv with public weights is local and draws no mask, so
// with public weights only the mask-only forward (A2B_BAKE_MASK_PASS) gives a residual sum its committed mask.
inline bool msb_input_residual() { return g_msb_input_residual && PUBLIC_WEIGHTS == 0; }

// RESHARE_OPT_SIM skips the reshare pre-send only where the bake guarantees it would be zero. Elsewhere the
// adder takes the real reshare: for inputs that were not baked, and for PPA/PPA4 under MWK with SecureML
// truncation but no cut, whose high reshared slices lie outside the truncated mask's image.
#define RESHARE_BAKE_COMPLETE                                                                         \
    (!(MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1 && TRUNC_DELAYED == 0 && (PPA_MSB == 1 || PPA4_MSB == 1) && \
       !CUT_FRAC_ELIGIBLE))
inline bool reshare_sim_on()
{
#if RESHARE_BAKE_ACTIVE && RESHARE_BAKE_COMPLETE  // #if: RESHARE_OPT may be undefined (A_KNOWN_TO_EVALUATORS_OPT)
    return msb_input_baked();
#else
    return false;
#endif
}

// A2B_ONLINE_OPT conv-mask bake (A2B_CONV_BAKE). Root problem it fixes: A2B_ONLINE_OPT precomputes the
// A2B S2 boolean share [c] = bool(-lv) via an interactive boolean addition of each party's bool(-lv_i).
// Two independent things must line up for the online msb adder to be correct:
//   (1) the conv mask lv committed in the PRE mask/send must equal the one in the LIVE mask/send, else
//       s1 = bool(mv = v+lv_live) and s2 = [c] = bool(-lv_pre) don't cancel; and
//   (2) the msb adder's beaver triples are generated in PRE from the s2 wire mask (out.l). If PRE sets
//       out.l to the boolean-adder INPUT (ia) while LIVE sets it to the OUTPUT [c], the triples are
//       generated for the wrong mask -> garbage. So PRE and LIVE must BOTH use [c] for out.l.
// The bake satisfies both by choosing, per party and BEFORE either FUNCTION pass, a random boolean A2B
// mask ia; deriving the conv mask lz = -untranspose(ia) (so ortho(-lz) == ia); running the boolean
// addition [c] = ia0 (+) ia1 = bool(-lz) EARLY (same stage as the LXLY triples); and then handing lz to
// every conv mask/send and [c] to every A2B-S2 slice in BOTH phases. g_a2b_ia -> boolean-adder input;
// g_a2b_lz -> conv mask; g_a2b_c -> [c] share consumed by prepare_A2B_S2.
// Multi-batch (DATTYPE > BITLENGTH) is covered, MODELWEIGHTS_KNOWN_DURING_PREPROCESSING too (its prescribed triple
// shares, mwk_choose_r1_*, are computed lane by lane).
// The bake commits one conv mask per full-width A2B slice group: with a reduced ReLU range (COMPRESS: bits
// REDUCED_BITLENGTH_m..k only) the boolean addition covers k - m of the 32 slices, so most outputs would have no
// committed mask. Full-width ReLUs only; COMPRESS runs the A2B unbaked.
#define A2B_CONV_BAKE_ACTIVE (A2B_ONLINE_OPT == 1 && A2B_CONV_BAKE == 1 && \
                              REDUCED_BITLENGTH_m == 0 && REDUCED_BITLENGTH_k == BITLENGTH)
// TS1 (reduced-slack truncation, TRUNC_APPROACH 1 / 4) in 2PC: a delayed ReLU input is truncated inside the ReLU's bit
// injection, from a public function of its masked value and preprocessed functions of its mask, which the A2B bake's
// Boolean addition provides (see g_ts1_la). No online message of its own.
#ifndef TS1_FOLD_POOL
#define TS1_FOLD_POOL (TRUNC_APPROACH == 1)  // a pooling fused into a TS1 ReLU divides in the TS1 lift (else TS{L})
#endif
#if TS1_FUSED_ACTIVE  // (TS1_FUSED_ACTIVE: generate_beaver_tiples.hpp)
#if TRUNC_DELAYED == 0
#error "TRUNC_APPROACH 1 / 4 with PROTOCOL 4: the truncation is fused into the ReLUs, set TRUNC_DELAYED=1"
#endif
#if !A2B_CONV_BAKE_ACTIVE
#error "TRUNC_APPROACH 1 / 4 with PROTOCOL 4: TS1 takes its mask carries from the A2B bake (A2B_ONLINE_OPT=1, A2B_CONV_BAKE=1, COMPRESS=0)"
#endif
#if OPTIMIZED_BIT_INJECTION_RELU == 0 || BIT_INJECTION_PREPROCESSING_OPT == 0 || ROT_PREPROCESSING_OPT == 0 || \
    FAKE_TRIPLES == 1 || FRACTIONAL < 1 || FRACTIONAL > BITLENGTH - 3
#error "TRUNC_APPROACH 1 / 4 with PROTOCOL 4: needs OPTIMIZED_BIT_INJECTION_RELU, BIT_INJECTION_PREPROCESSING_OPT, ROT_PREPROCESSING_OPT, real triples"
#endif
#endif
// A2B_DELAYED_CUT for TS{L} (config.h): a delayed ReLU input's A2B converts the locally truncated value, with the cut
#define A2B_DCUT_ACTIVE (TRUNC_DELAYED == 1 && A2B_DCUT_ELIGIBLE && A2B_CONV_BAKE_ACTIVE && CUT_FRAC_ELIGIBLE)
// ... without the bake (A2B_ONLINE_OPT=0, plain or reshared): the A2B converts the parties' locally shifted additive
// shares, (m - l_0) >> F and (-l_1) >> F (g_a2b_share_shift, prepare_A2B_S1 / S2 in every pass)
#define A2B_DCUT_SHARE_ACTIVE (TRUNC_DELAYED == 1 && A2B_DCUT_ELIGIBLE && A2B_ONLINE_OPT == 0 && CUT_FRAC_ELIGIBLE)
// A BatchNorm with secret parameters re-masks its output (BN triples, SecureML truncation leaves the drawn mask, beta's
// mask is added afterwards like a conv's bias), so it can take the committed / reshare-baked masks like a conv/FC.
// Not with FUSE_CONV_BN: then every BatchNorm forward passes its input on (also one that follows a pooling layer).
#define BN_BAKE_SUPPORTED (PROTOCOL == 4 && BN2D_TRIPLES == 1 && PUBLIC_WEIGHTS == 0 && TRUNC_DELAYED == 0 && \
                           TRUNC_APPROACH == 0 && FUSE_CONV_BN_SIM == 0 && FUSE_CONV_BN == 0 && ROT_PREPROCESSING_OPT == 1 && \
                           A2B_BAKE_BN == 1 && (A2B_CONV_BAKE_ACTIVE || RESHARE_BAKE_ACTIVE))
// A2B_BAKE_MASK_PASS (public weights, single batch): no ReLU input can be baked by its producer (a public-weight conv,
// pooling or BatchNorm fixes the mask as a linear function of earlier bit-injection masks), so instead the
// preprocessing pass first runs the network over the masks alone: ReLUs record their input masks lambda_v and
// output committed bit-injection masks (g_relu_out), which the real passes use as well. The Boolean addition then
// runs on bool(-lambda_v) itself, and no ReLU input needs the rebase message.
#define A2B_MASK_PASS_ACTIVE (A2B_CONV_BAKE_ACTIVE && A2B_BAKE_MASK_PASS == 1 && PUBLIC_WEIGHTS == 1 && DATTYPE == BITLENGTH && \
                              OPTIMIZED_BIT_INJECTION_RELU == 1 && (TRUNC_APPROACH == 0 || TS1_FUSED_ACTIVE) && \
                              TRUNC_DELAYED == 1 && RANDOM_ALGORITHM == 2 && USE_SSL_AES == 0)
// CHEETAH_CONV_EARLY (secret weights, packed + pipelined convs, single batch): the preprocessing pass first runs the
// network over the masks alone; the convs record their triples' inputs (every conv input is a ReLU output, whose mask
// is committed, or the network input), and the conv triples start on channels of their own while the OT phase runs
// (instead of alongside the pass). The OT phase runs from within the pass, after the mask-only forward.
#define CHEETAH_CONV_EARLY_ACTIVE (CHEETAH_CONV_ASYNC_ACTIVE && CHEETAH_CONV_EARLY == 1 && PUBLIC_WEIGHTS == 0 && \
                                   CONV_TRIPLES == 1 && ROT_PREPROCESSING_OPT == 1 && OPTIMIZED_BIT_INJECTION_RELU == 1 && \
                                   (TRUNC_APPROACH == 0 || TS1_FUSED_ACTIVE) && RANDOM_ALGORITHM == 2 && USE_SSL_AES == 0)
#define MASK_FORWARD_ACTIVE (A2B_MASK_PASS_ACTIVE || CHEETAH_CONV_EARLY_ACTIVE)
// The slot of the current bit injection's first output (MASK_FORWARD_ACTIVE, set per element by
// bit_injection_opt_range inside a ReLU; -1: a fresh mask)
inline thread_local int64_t tl_bi_slot = -1;
#if MASK_FORWARD_ACTIVE
// Every ReLU's outputs have committed masks (bit injection), by slot: a ReLU of len values takes len rounded up to
// BITLENGTH slots, from g_relu_base on, which restarts with every forward. Counted in the INIT pass.
inline std::vector<DATATYPE> g_relu_out;
inline uint64_t g_relu_slots = 0;
inline uint64_t g_relu_base = 0;
inline uint64_t g_bi_base = UINT64_MAX;  // the running ReLU's first slot, for its bit injection

[[noreturn]] inline void mask_pass_abort(const char* what)
{
    fprintf(stderr, "mask-only forward: %s\n", what);
    std::abort();
}
#endif
#if RANDOM_ALGORITHM == 2 && USE_SSL_AES == 0
// Counter-mode values under this party's PSELF key, by (tweak, index): random access, and independent of the
// generator stream (PSELF), whose position after a ReLU differs between the mask-only forward and the real one.
inline DATATYPE prf_value(uint64_t tweak, uint64_t k)
{
    static thread_local uint64_t cached_tweak = 0, cached = UINT64_MAX;
    alignas(sizeof(AES_TYPE)) static thread_local DATATYPE buf[BUFFER_SIZE];
    const uint64_t blk = k / BUFFER_SIZE;
    if (blk != cached || tweak != cached_tweak)
    {
        alignas(sizeof(AES_TYPE)) uint64_t in[sizeof(AES_TYPE) / 8];
        for (size_t i = 0; i < sizeof(AES_TYPE) / 8; i++) in[i] = tweak ^ ((uint64_t) i << 56) ^ blk;
        AES_TYPE st;
        std::memcpy(&st, in, sizeof(st));
        AES_enc(st, key_schedule[PSELF]);
        MM_AES_STORE((AES_TYPE*) buf, st);
        cached = blk, cached_tweak = tweak;
    }
    return buf[k % BUFFER_SIZE];
}
// prf_value's values must not repeat: until BUFFER_SIZE was parenthesized, k / BUFFER_SIZE divided by AES_DATTYPE and
// then by DATTYPE, and each value came back for 32 consecutive k (the committed masks masked many values each)
inline void prf_value_check()
{
    int same = 0;
    for (uint64_t k = 1; k < 4 * BUFFER_SIZE + 8; k++)
    {
        const DATATYPE a = prf_value(0x1234, k - 1), b = prf_value(0x1234, k);
        same += std::memcmp(&a, &b, sizeof(DATATYPE)) == 0;
    }
    if (same > 1)
    {
        fprintf(stderr, "prf_value: %d of its neighbouring values are equal\n", same);
        std::abort();
    }
}
constexpr uint64_t kTweakTrunc = 0x5bd1e995a2b0c0deULL, kTweakRelu = 0x2545f4914f6cdd1dULL,
                   kTweakResidual = 0x7a3d2c1b0f9e8d7cULL, kTweakA2bIa = 0x3c6ef372fe94f82bULL;
#endif
#if A2B_CONV_BAKE_ACTIVE
#ifndef A2B_BAKE_INIT_THREADS
#define A2B_BAKE_INIT_THREADS 16  // threads deriving the committed masks (init_a2b_bake), in the OT phase
#endif
std::vector<DATATYPE> g_a2b_ia;   // this party's random boolean A2B-mask slices (boolean-adder input)
std::vector<DATATYPE> g_a2b_lz;   // derived conv mask -untranspose(ia); ortho(-lz) == ia
std::vector<DATATYPE> g_a2b_c;    // this party's share of [c] = bool(-lz), from the early boolean addition
uint64_t g_a2b_layer_base = 0;    // g_a2b_lz base for the current layer's A2B group (reset per phase)
uint64_t g_a2b_c_cursor = 0;      // A2B-S2 [c] cursor (reset per phase)
// Shifted A2B slots (TS1's shifted design, A2B_DCUT_ACTIVE): the Boolean addition adds the parties' mask shares times a
// public factor and shifted right by FRACTIONAL, (factor nu_i mod 2^l) >> F, instead of nu_i. Its low l - F sum bits
// are then the Boolean form of the locally truncated value's mask, which the ReLU's A2B converts with the cut.
// Recorded by the INIT pass, in forward order.
struct A2bShiftRange
{
    uint64_t slot_base, slots;
    UINT_TYPE factor;
};
inline std::vector<A2bShiftRange> g_a2b_shift_ranges;
inline void a2b_record_shift(uint64_t slot_base, uint64_t slots, UINT_TYPE factor)
{
    g_a2b_shift_ranges.push_back({slot_base, slots, factor});
}
// This party's share of nu, value by value, of the group of BITLENGTH slices at ia (the A2B's slice layout)
inline void a2b_group_values(const DATATYPE* ia, UINT_TYPE* nu)
{
    DATATYPE t[BITLENGTH];
    for (int i = 0; i < BITLENGTH; i++) t[i] = ia[i];
    unorthogonalize_boolean(t, nu);
}
// The shifted input of a value: (factor nu_i mod 2^l) >> F
inline UINT_TYPE a2b_shifted(UINT_TYPE nu, UINT_TYPE factor)
{
    return (UINT_TYPE) (factor * nu) >> FRACTIONAL;
}
// Replace this party's Boolean-addition inputs of the shifted slots (loaded from g_a2b_ia) by the shifted values
inline void a2b_shift_inputs(DATATYPE* in)
{
    for (const A2bShiftRange& r : g_a2b_shift_ranges)
        for (uint64_t base = r.slot_base; base < r.slot_base + r.slots; base += BITLENGTH)
        {
            alignas(sizeof(DATATYPE)) UINT_TYPE nu[DATTYPE];
            a2b_group_values(&g_a2b_ia[base], nu);
            for (int j = 0; j < DATTYPE; j++) nu[j] = a2b_shifted(nu[j], r.factor);
            orthogonalize_boolean(nu, &in[base]);
        }
}
// init_a2b_bake / a2b_bake_store_c are defined further down, after the boolean_addition_triple buffers.

// Index-addressed (NOT a linear cursor): the conv mask for the e-th layer-local output goes to
// g_a2b_lz[layer_base + g_bake_batch_offset + e]. e is the C[] output position within the batch
// element (mask/send index / bake_index), g_bake_batch_offset the batch-element base (set by the
// conv/FC forward), g_a2b_layer_base the layer base (snapped to the [c] group boundary after each
// A2B). This matches the A2B, which packs C[] linearly into sints and consumes [c] slice-per-position,
// even when the conv mask/send is called in tiled (non-linear) order.
// The committed mask of A2B slot e of the current layer (a ReLU input's rebase target; bias-compensated for a conv/FC).
template <typename Datatype, typename func_sub>
inline Datatype a2b_bake_slot_mask(uint64_t e, func_sub SUB)
{
    const uint64_t idx = g_a2b_layer_base + g_bake_batch_offset + e;
    // Out of range == this output never feeds an A2B (g_a2b_lz covers exactly the INIT-counted A2B
    // slices; e.g. the network's final FC before the reveal). Fall back to a fresh synced PRNG draw -
    // the baseline behavior. Returning a constant here instead would make P1's r1 = -low TINY and
    // break the SecureML trunc wrap (B >= |v| fails) -> every negative output off by +2^(K-F).
    if (idx >= g_a2b_lz.size())  // also the INIT pass, which runs before commitment and must not draw
        return current_phase == PHASE_INIT ? SET_ALL_ZERO() : getRandomVal(PSELF);
    Datatype lz = g_a2b_lz[idx];
    // Bias pre-compensation: add_bias later shifts this output's mask by the party's OWN bias-mask
    // share (owner: get_mask() of its b share; non-owner: 0 -> no-op, so P1's trunc-image constraint
    // is untouched). Subtract it here so the TOTAL mask after add_bias equals the committed lz that
    // [c] was built for. Same buffer/indexing as the reshare bake (g_bake_bias_l, batch-local e).
    if (g_bake_bias_l != nullptr && g_bake_bias_len > 0)
        lz = SUB(lz, g_bake_bias_l[((g_bake_batch_offset + e) % g_bake_bias_len) / g_bake_bias_rep]);
    return lz;
}

// P1's conv/FC output masks lie in the SecureML truncation's image (top FRACTIONAL bits zero) when it prescribes its
// triple share (weights known in preprocessing, a known, TRUNC_DELAYED=0: l1 = TRUNC(-r1)). Everywhere else the masks
// are free draws: the committed masks must then be uniform on the whole ring, since the other party sees the masked
// share value (a narrowed mask would reveal part of it).
#define A2B_P1_IMAGE_MASKS (PARTY == 1 && TRUNC_DELAYED == 0 && A_KNOWN == 1)
// Residual partners: every party subtracts the other addend's mask. P1 with image masks can only if the other addend's
// mask is committed as well (A2B_RESIDUAL_COMMIT, see g_residual_sums): then lz_1 = m_a + m_b and the partner draws
// m_a. Otherwise P1 draws fresh and moves the sum with a one-sided rebase (rebase_p1); drawing lz there would reuse
// the slot through the rebase.
#define A2B_RESIDUAL_BAKE_P1 (MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 0 || A_KNOWN == 0 || TRUNC_DELAYED == 1)
#define A2B_RESIDUAL_COMMIT (!A2B_RESIDUAL_BAKE_P1 && A2B_BAKE_RESIDUAL == 1 && DATTYPE == BITLENGTH && \
                             RANDOM_ALGORITHM == 2 && USE_SSL_AES == 0)

// Is P1's part of residual sum k committed (no rebase_p1)? The same answer at both parties: from the network's marks
// (the producer kind) alone. A ReLU producer's masks are committed only with a mask-only forward (g_relu_out).
inline bool a2b_residual_committed(int k)
{
#if A2B_RESIDUAL_COMMIT
    if (k < 0 || (size_t) k >= g_residual_sums.size())
        return false;
    const int p = g_residual_sums[k].producer;
    return p == 1 || (p == 2 && MASK_FORWARD_ACTIVE);
#else
    (void) k;
    return false;
#endif
}

#if A2B_RESIDUAL_COMMIT && PARTY == 1
// P1's committed mask m_b of value j of residual sum k's other addend: a conv/FC's is a PRF value in the truncation's
// image (uniform there, like a fresh m1: the prescribed r1 = -((m_b << F) + low) stays uniform), a ReLU's is its
// committed bit-injection output mask.
inline DATATYPE a2b_residual_other_mask(int k, uint64_t j)
{
    const ResidualSum& r = g_residual_sums[k];
#if MASK_FORWARD_ACTIVE
    if (r.producer == 2)
        return r.relu_base + j < g_relu_out.size() ? g_relu_out[r.relu_base + j] : SET_ALL_ZERO();
#endif
    const UINT_TYPE v = (UINT_TYPE) prf_value(kTweakResidual, ((uint64_t) k << 40) | j);
    return (DATATYPE) (v & (((UINT_TYPE) 1 << (BITLENGTH - FRACTIONAL)) - (UINT_TYPE) 1));
}
#endif

// A conv/FC output's mask: the committed slot mask when the layer feeds a baked ReLU (g_conv_bake), a fresh
// synced draw otherwise, so that every committed slot masks exactly one value (see g_conv_bake).
template <typename Datatype, typename func_sub>
inline Datatype a2b_bake_conv_mask(uint64_t e, func_sub SUB)
{
#if A2B_RESIDUAL_COMMIT && PARTY == 1
    // the other addend of a residual sum whose P1 part is committed: its committed mask m_b
    if (g_res_producer_k >= 0 && a2b_residual_committed(g_res_producer_k))
        return current_phase == PHASE_INIT ? SET_ALL_ZERO()
                                           : (Datatype) a2b_residual_other_mask(g_res_producer_k, g_bake_batch_offset + e);
#endif
    if (!g_conv_bake)
        return current_phase == PHASE_INIT ? SET_ALL_ZERO() : getRandomVal(PSELF);
    if (g_bake_res_l != nullptr)
    {
#if PARTY == 1 && !A2B_RESIDUAL_BAKE_P1
        if (!a2b_residual_committed(g_bake_res_k))
            return current_phase == PHASE_INIT ? SET_ALL_ZERO() : getRandomVal(PSELF);
#endif
        if (current_phase == PHASE_INIT)
            return SET_ALL_ZERO();
        return SUB(a2b_bake_slot_mask<Datatype>(e, SUB), (Datatype) g_bake_res_l[g_bake_batch_offset + e]);
    }
    return a2b_bake_slot_mask<Datatype>(e, SUB);
}

// [c] share for the next A2B-S2 slice - identical in PRE and LIVE, so the msb adder's beaver triples
// (generated in PRE from this out.l) match what LIVE consumes.
// The value being prepared reads its slices from tl_a2b_c on (set per value by get_msb_range, so that the values are
// prepared on the pool); -1: the serial cursor.
inline thread_local int64_t tl_a2b_c = -1;
// TS1_CUT_ACTIVE: the ReLU converts trunc(z), whose mask's Boolean form is the bake's [c] of z shifted by FRACTIONAL
// bits: slice i (i >= shift; 0 = MSB) of the value's group is slice i - shift of [c] (the top `shift` slices are cut)
inline int g_a2b_c_shift = 0;
inline DATATYPE a2b_bake_get_c()
{
    if (tl_a2b_c >= 0)
    {
        const uint64_t at = (uint64_t) tl_a2b_c++;
        if (g_a2b_c_shift > 0)
        {
            if ((int) (at % BITLENGTH) < g_a2b_c_shift)
                return SET_ALL_ZERO();  // a vacant slice (the cut replaces it by 0 anyway)
            return at - g_a2b_c_shift < g_a2b_c.size() ? g_a2b_c[at - g_a2b_c_shift] : SET_ALL_ZERO();
        }
        return at < g_a2b_c.size() ? g_a2b_c[at] : SET_ALL_ZERO();
    }
    return (g_a2b_c_cursor < g_a2b_c.size()) ? g_a2b_c[g_a2b_c_cursor++] : SET_ALL_ZERO();
}
#endif

// PPA4 comm-elimination thresholds: a gate/send/zero_add site with threshold T is skipped when
// FRACTIONAL >= T (its g-factor coverage is then entirely identity-substituted -> public output).
// The P-gate beaver3 slots (skipped in ALL phases, so allocation and retrieval both drop) shift the
// consumption RANK of all later slots within each adder - external offset arithmetic (the S1 peek
// and the bake) must use the cut-aware count and ranks below.
constexpr int cut_frac_ppa4_b3_pslot_th(int slot)  // -1 = not a P-slot (never skipped)
{
    switch (slot)
    {
        case 1: return 3; case 3: return 6; case 5: return 9; case 7: return 12; case 9: return 15;
        case 11: return 18; case 13: return 21; case 15: return 25; case 17: return 28; case 20: return 9;
        default: return -1;
    }
}
constexpr int cut_frac_ppa4_b3_skipped_below(int slot)
{
#if CUT_FRAC_ELIGIBLE_PPA4
    int n = 0;
    for (int j = 0; j < slot; j++)
    {
        const int th = cut_frac_ppa4_b3_pslot_th(j);
        if (th >= 0 && FRACTIONAL >= th)
            n++;
    }
    return n;
#else
    (void) slot;
    return 0;
#endif
}
constexpr bool cut_frac_ppa4_skip(int thresholdF)
{
#if CUT_FRAC_ELIGIBLE_PPA4
    return FRACTIONAL >= thresholdF;
#else
    (void) thresholdF;
    return false;
#endif
}

// Slice roles when the cut is active (adder width k == BITLENGTH; slice 0 = numeric MSB):
//  - slices [0, FRACTIONAL):  vacant - never prepared, shared, reshared, or read.
//  - slice FRACTIONAL:        boundary - its RAW wire pair is kept (masked + sent in the A2B
//                             prepare, taking over slice 0's original role) because the tree's
//                             output tap p_0 is substituted by a[FRACTIONAL] ^ b[FRACTIONAL];
//                             its LEAF values are still identity-substituted (g := 0, p := 1).
//  - slices (FRACTIONAL, k):  unchanged.
// The identity substitution (g_i, p_i) := (public 0, public 1) for slices 1..FRACTIONAL makes the
// UNCHANGED prefix tree compute p_F ^ G(F+1 .. k-1) - the reduced-width MSB - because identity
// elements drop out of every prefix combine (verified by exhaustive simulation).
constexpr bool cut_frac_vacant(int k, int i)  // fully-skipped slice?
{
#if CUT_FRAC_ELIGIBLE && !CUT_FRAC_NARROW
    return k == BITLENGTH && i < FRACTIONAL;
#else
    (void) k; (void) i;
    return false;
#endif
}

constexpr bool cut_frac_identity(int k, int i)  // leaf (g,p) := (0,1) substituted slice?
{
#if CUT_FRAC_ELIGIBLE && !CUT_FRAC_NARROW
    return k == BITLENGTH && i >= 1 && i <= FRACTIONAL;
#else
    (void) k; (void) i;
    return false;
#endif
}

// Runtime slice-role helpers for the A2B prepare/complete loops (the flag distinguishes ReLU
// adders, which apply the cut, from max/min/comparison adders on the same build, which don't).
// prepare_A2B_* receives a slice RANGE (m, k); the cut only applies to full-width conversions.
inline bool cut_frac_prep_vacant(int m, int k, int i)
{
#if CUT_FRAC_ELIGIBLE
    return g_cut_frac_active && m == 0 && k == BITLENGTH && i < FRACTIONAL;
#else
    (void) m; (void) k; (void) i;
    return false;
#endif
}
// Constructor-side reshare skip for identity slices.
inline bool cut_frac_skip_reshare(int k, int i)
{
#if CUT_FRAC_ELIGIBLE
    return g_cut_frac_active && cut_frac_identity(k, i);
#else
    (void) k; (void) i;
    return false;
#endif
}

inline bool cut_frac_prep_boundary(int m, int k, int i)
{
#if CUT_FRAC_ELIGIBLE
    return g_cut_frac_active && m == 0 && k == BITLENGTH && i == FRACTIONAL;
#else
    (void) m; (void) k; (void) i;
    return false;
#endif
}

// BITLENGTH 64 (CUT_FRAC_NARROW): the A2B prepares the full width as above (vacant slices public constants, the
// boundary slice masked and sent), and the MSB adder of width BITLENGTH - FRACTIONAL (narrow64/) runs on slices
// FRACTIONAL..BITLENGTH-1 (get_msb_range): the MSB of the low BITLENGTH - FRACTIONAL bits of the sum, the sign of the
// value. The full-width circuit is not identity-substituted (cut_frac_vacant / cut_frac_identity stay false).
inline bool cut_frac_narrow_on(int m, int k)
{
#if CUT_FRAC_NARROW
    return g_cut_frac_active && m == 0 && k == BITLENGTH;
#else
    (void) m; (void) k;
    return false;
#endif
}
// The reshared four-way circuit's input slices, by position in the prepared range [m, k): with the narrow cut, those
// of the narrow adder (i >= FRACTIONAL; the vacant and boundary slices are handled first)
inline bool ppa4_reshared_at(int m, int k, int i)
{
    if (cut_frac_narrow_on(m, k))
        return i > FRACTIONAL && is_ppa4_reshared(k - FRACTIONAL, i - FRACTIONAL);
    return is_ppa4_reshared(k - m, i - m);
}

// BITLENGTH 64: the reshared four-way circuit the bake serves is the full-width one, or with the narrow cut the one of
// 64 - FRACTIONAL bits on slices FRACTIONAL.. (slice i -> i - FRACTIONAL); its maps (gen_64bit_adders.py checks them
// against the generated circuits): reshared slices j % 3 == 1 with rt[(j - 1) / 3], SIM-skipped zero_adds at
// j % 3 == 2 with 3-tuple 2 (j - 2) / 3 where ppa4_reshared_sim_za_wide (narrow64/widths.h)
constexpr int wide_ppa4_width() { return CUT_FRAC_NARROW ? BITLENGTH - FRACTIONAL : BITLENGTH; }
constexpr int wide_ppa4_slice(int i) { return CUT_FRAC_NARROW ? i - FRACTIONAL : i; }

// Reshare wiring of the *_and_ab_reshared adders: which bit-slice (adder wire index i, 0 = numeric MSB,
// k-1 = numeric LSB) is reshared with which random_triples[] offset within one adder. -1 = not reshared.
// Must mirror the generated circuit constructors (rca_msb / ppa_msb_unsafe / ppa_msb_4way _and_ab_reshared.hpp).
constexpr int reshare_rt_offset(int k, int i)
{
#if RCA_MSB == 1
    return (i == k - 1) ? 0 : -1;  // RCA reshares only the LSB slice (first carry gate), rt[0]
#elif PPA_MSB == 1
#if CUT_FRAC_ELIGIBLE
    // CUT: slices 1..FRACTIONAL are identity-substituted (not reshared, no rt consumed); kept
    // slices consume sequentially, so slice i's offset shifts down by FRACTIONAL.
    return (i >= FRACTIONAL + 1 && i < k) ? i - 1 - FRACTIONAL : -1;
#else
    return (i >= 1 && i < k) ? i - 1 : -1;  // PPA reshares slices 1..k-1 with rt[i-1], ascending
#endif
#elif PPA4_MSB == 1
    // PPA4 reshares the AND2 "generate" wires; retrieval order is circuit-specific (k=32: wire 22 is LAST).
    if (k == 64)
    {
        const int w = wide_ppa4_width(), j = wide_ppa4_slice(i);
        return (j >= 1 && j < w && j % 3 == 1) ? (j - 1) / 3 : -1;
    }
    if (k == 32)
    {
#if CUT_FRAC_ELIGIBLE_PPA4
        // CUT: identity-substituted slices (1..FRACTIONAL) are not reshared; kept slices consume
        // sequentially by RANK among kept slices in the retrieval order 1,4,7,...,29,22.
        {
            constexpr int order[11] = {1, 4, 7, 10, 13, 16, 19, 23, 26, 29, 22};
            int rank = 0;
            for (int j = 0; j < 11; j++)
            {
                if (order[j] <= FRACTIONAL)
                    continue;  // skipped (identity)
                if (order[j] == i)
                    return rank;
                rank++;
            }
            return -1;
        }
#else
        switch (i)
        {
            case 1: return 0; case 4: return 1; case 7: return 2; case 10: return 3; case 13: return 4;
            case 16: return 5; case 19: return 6; case 23: return 7; case 26: return 8; case 29: return 9;
            case 22: return 10;
            default: return -1;
        }
#endif
    }
    else if (k == 16)
    {
        switch (i)
        {
            case 1: return 0; case 4: return 1; case 7: return 2; case 10: return 3; case 13: return 4;
            default: return -1;
        }
    }
    else if (k == 8)
    {
        switch (i)
        {
            case 2: return 0; case 5: return 1; case 1: return 2;
            default: return -1;
        }
    }
    return -1;
#else
    return -1;
#endif
}

// Random multiplications consumed by one MSB adder of width k.
constexpr uint64_t reshares_per_adder(int k)
{
#if RCA_MSB == 1
    return 1;
#elif PPA_MSB == 1
#if CUT_FRAC_ELIGIBLE
    return (uint64_t)(k - 1 - FRACTIONAL);  // CUT: identity slices consume no rt
#else
    return (uint64_t)(k - 1);
#endif
#elif PPA4_MSB == 1
    if (k == 64)
        return (uint64_t) ((wide_ppa4_width() - 2) / 3 + 1);
#if CUT_FRAC_ELIGIBLE_PPA4
    if (k == 32)
    {
        constexpr int order[11] = {1, 4, 7, 10, 13, 16, 19, 23, 26, 29, 22};
        uint64_t n = 0;
        for (int j = 0; j < 11; j++)
            if (order[j] > FRACTIONAL)
                n++;
        return n;
    }
    return k == 16 ? 5 : 3;
#else
    return k == 32 ? 11 : (k == 16 ? 5 : 3);
#endif
#else
    return 0;
#endif
}

// PPA4 SIM=1 input-wire zero_adds: slice i's a-wire is re-masked to beaver3_tuples[t].b and its
// b-wire to beaver3_tuples[t].c (per-adder tuple index t; -1 = no gated zero_add on this slice).
// Extracted from ppa_msb_4way_and_ab_reshared.hpp (the RESHARE_OPT_SIM == 1 branches).
constexpr int ppa4_zero_add_t3(int k, int i)
{
    if (k == 64)
    {
        const int w = wide_ppa4_width(), j = wide_ppa4_slice(i);
        return (ppa4_reshared_sim_za_wide(w) && j >= 2 && j < w && j % 3 == 2) ? 2 * (j - 2) / 3 : -1;
    }
    if (k == 32)
    {
        int slot = -1;
        switch (i)
        {
            case 2: slot = 0; break; case 5: slot = 2; break; case 8: slot = 4; break;
            case 11: slot = 6; break; case 14: slot = 8; break; case 17: slot = 10; break;
            case 20: slot = 12; break; case 24: slot = 14; break; case 27: slot = 16; break;
            case 30: slot = 18; break;
            default: return -1;
        }
        return slot - cut_frac_ppa4_b3_skipped_below(slot);  // consumption rank under the cut
    }
    else if (k == 16)
    {
        switch (i)
        {
            case 2: return 0; case 5: return 2; case 8: return 4; case 11: return 6;
            case 14: return 7;
            default: return -1;
        }
    }
    else if (k == 8)
    {
        switch (i)
        {
            case 3: return 0; case 6: return 2;
            default: return -1;
        }
    }
    return -1;
}

// Beaver 3-tuples consumed by one PPA4 MSB adder of width k (Beaver3TupleCount in the circuit).
// Under the cut, skipped P-gate slots consume nothing (retrieval and INIT allocation both skip).
constexpr uint64_t b3_tuples_per_adder(int k)
{
    if (k == 64)
        return (uint64_t) ppa4_reshared_b3_count_wide(wide_ppa4_width());
    if (k == 32)
        return (uint64_t)(24 - cut_frac_ppa4_b3_skipped_below(24));
    return k == 16 ? 9 : 4;
}

// P0-side helper for the PPA4 SIM=1 zero_add skip: counts prepare_A2B_S1 calls since the last
// beaver-3-tuple retrieval, so slice masks can be peeked at the tuple positions the group's adder
// WILL consume (all groups are prepared before any adder is constructed). Reset in
// retrieveBeaver3Tuple (any retrieval means the S1 batch has ended), and by the ReLU's A2B right after
// its prepare level. Inside a parallel circuit level each worker counts on its own cursor (IDX_A2B_S1_PENDING).
uint64_t g_a2b_s1_pending = 0;


// RESHARE_OPT_SIM: bake the reshare random multiplication rt.a into P1's (negated) conv-output mask -l, so the ReLU's
// A2B b-input bool(-l) already equals P1's rt.a share at every reshared bit-slice -> the skipped reshare_b
// preprocessing send (delta = l ^ rt.a) would be 0, making the SIM=1 execution bit-identical to SIM=0.
// MUST be called IDENTICALLY in PRE and online (l is PRNG-synced across phases, and the adders consume random
// multiplications at the same sequence points in both phases). bake_index = the value's LAYER-LOCAL linear output
// index e (the GEMM masks in tiled order, so we key off this, not call order).
//
// Mapping (verified against real_ortho): the A2B transposes BITLENGTH values into slices with BOTH indices mirrored:
// value j (= e % BITLENGTH), numeric bit b -> slice (BITLENGTH-1-b), lane-bit (BITLENGTH-1-j). The adder for group
// g (= e / BITLENGTH) consumes random_multiplication_a[base + g*R + t] where base = curr_random_multiplication_index
// at conv-mask time (nothing else consumes between the conv and its ReLU A2B) and t = reshare_rt_offset(k, slice).
// The baked slices of one adder of width k as (numeric bit, tuple offset) pairs, fixed at compile time: the
// per-slice helpers (the PPA4 cut ranks scan the retrieval order) ran for all k slices of every conv output.
struct BakeSlices
{
    int n = 0;
    int nb[BITLENGTH] = {};
    int t[BITLENGTH] = {};
};
constexpr BakeSlices bake_slices(int k, bool b3)
{
    BakeSlices s;
    for (int i = 1; i < k; i++)  // slice 0 (numeric MSB) is never reshared
    {
        if (cut_frac_identity(k, i))
            continue;  // CUT_FRACTIONAL_BITS_OPT: identity-substituted slice, neither reshared nor zero_added
        const int t = b3 ? ppa4_zero_add_t3(k, i) : reshare_rt_offset(k, i);
        if (t < 0)
            continue;
        s.nb[s.n] = k - 1 - i;  // numeric bit position of slice i
        s.t[s.n++] = t;
    }
    return s;
}

template <typename Datatype, typename func_sub>
inline void bake_reshare_mask(Datatype& l, int bake_index, func_sub SUB)
{
#if PARTY == 1 && RESHARE_BAKE_ACTIVE  // P1-ONLY: baking P0's mask online would desync its PRE vs LIVE masks
    constexpr int K = BITLENGTH;
    constexpr uint64_t R = reshares_per_adder(K);
    static constexpr BakeSlices rs = bake_slices(K, false);
    const uint64_t e = (uint64_t) bake_index + g_bake_batch_offset;  // batch-global output index
    const uint64_t g = e / K;        // bit-sliced A2B group (one adder per group)
    const int j = (int) (e % K);     // value's word index in the group -> lane-bit (K-1-j) after the transpose
    const uint64_t base = curr_random_multiplication_index + g * R;
    if (base + R > num_random_multiplications)
        return;  // this layer's outputs never reach an MSB adder (e.g. final layer) - leave the mask random
    UINT_TYPE negl = (UINT_TYPE) l;
    for (int s = 0; s < rs.n; s++)
    {
        const UINT_TYPE rta = (UINT_TYPE) random_multiplication_a[base + (uint64_t) rs.t[s]];
        const UINT_TYPE bit = (rta >> (K - 1 - j)) & (UINT_TYPE) 1;
        negl = (negl & ~((UINT_TYPE) 1 << rs.nb[s])) | (bit << rs.nb[s]);
    }
#if PPA4_MSB == 1
    // PPA4 additionally SIM-skips the input-wire zero_adds: bake our (P1-local, see party_local_bc
    // in the tuple generation) beaver3 .c fields into the zero_added b-wire slices so the skipped
    // re-masking would have been a no-op. Same base-offset reasoning as the random multiplications.
    static constexpr BakeSlices zs = bake_slices(K, true);
    constexpr uint64_t B3 = b3_tuples_per_adder(K);
    const uint64_t b3_base = curr_beaver_3_triple_index + g * B3;
    if (b3_base + B3 <= num_beaver_3_tuples)
    {
        for (int s = 0; s < zs.n; s++)
        {
            const UINT_TYPE c3 = (UINT_TYPE) beaver_3_tuples.c[b3_base + (uint64_t) zs.t[s]];
            const UINT_TYPE bit = (c3 >> (K - 1 - j)) & (UINT_TYPE) 1;
            negl = (negl & ~((UINT_TYPE) 1 << zs.nb[s])) | (bit << zs.nb[s]);
        }
    }
#endif
    Datatype l_new = SUB(SET_ALL_ZERO(), (Datatype) negl);
    if (g_bake_bias_l != nullptr && g_bake_bias_len > 0)  // pre-compensate a shared bias added after the GEMM
        l_new = SUB(l_new, g_bake_bias_l[(e % g_bake_bias_len) / g_bake_bias_rep]);
    l = l_new;  // final ReLU-input mask == -negl => the A2B input -l transposes to rt.a at reshared slices
#else
    (void) l; (void) bake_index;
#endif
}

// MODELWEIGHTS_KNOWN + RESHARE_OPT_SIM, SecureML (non-delayed) truncation: construct P1's freely
// prescribed triple share r1 so that its output mask l = TRUNC(-r1) (LOGICAL shift by FRACTIONAL)
// carries the baked reshare bits: -r1 := (l_baked << FRACTIONAL) + low with low < 2^FRACTIONAL.
// Only mask bits 0..K-FRACTIONAL-1 are realizable (the trunc image zeroes the top FRACTIONAL bits),
// so reshared slices at numeric bits >= K-FRACTIONAL stay unbaked: exact for RCA (reshares bit 0
// only); PPA/PPA4 additionally need TRUNC_DELAYED=1 (see the without_trunc a_known variant).
template <typename Datatype, typename func_sub>
inline Datatype construct_mwk_r1_baked(Datatype r1_base, Datatype low_rand, int bake_index, func_sub SUB)
{
#if PARTY == 1 && RESHARE_BAKE_ACTIVE
    Datatype l_t = r1_base;
    bake_reshare_mask(l_t, bake_index, SUB);
    const UINT_TYPE low = (UINT_TYPE) low_rand & (((UINT_TYPE) 1 << FRACTIONAL) - (UINT_TYPE) 1);
    return (Datatype) (UINT_TYPE) (0 - (((UINT_TYPE) l_t << FRACTIONAL) + low));
#else
    (void) low_rand; (void) bake_index;
    return r1_base;
#endif
}

// P1's prescribed triple share for the a_known (MODELWEIGHTS_KNOWN) paths. Used by BOTH the PRE and
// the online phase so the PRNG draw sequences match by construction.
// SecureML-truncated mask l = TRUNC(-r1): bake image-limited to bits 0..K-FRACTIONAL-1 (RCA-exact).
template <typename Datatype, typename func_sub>
inline Datatype mwk_choose_r1_trunc(int bake_index, func_sub SUB)
{
#if A2B_CONV_BAKE_ACTIVE
    // A2B bake, TD=0: prescribe r1 so P1's SecureML-truncated mask l1 = TRUNC(-r1) == the committed
    // (sign-extended) mask m1 = a2b_bake_conv_mask. -r1 := (m1 << FRACTIONAL) + low, low < 2^FRACTIONAL
    // fresh: the low bits are truncated away, and m1's top FRACTIONAL bits are sign-extension so
    // (m1 << F) >> F == m1. The `low` PRNG draw is identical in PRE and LIVE (synced PSELF), so r1 (the
    // prescribed triple share) matches. [c] = bool(-(lz0+m1)) was formed from the same m1 in init.
    // Lane by lane (multi-batch: a Datatype holds DATTYPE / BITLENGTH values); one word: the same values as before.
    const Datatype m1 = a2b_bake_conv_mask<Datatype>((uint64_t)(bake_index < 0 ? 0 : bake_index), SUB);
    const Datatype low = OP_AND(getRandomVal(PSELF), PROMOTE(((UINT_TYPE) 1 << FRACTIONAL) - (UINT_TYPE) 1));
    return SUB(SET_ALL_ZERO(), OP_ADD(OP_SHIFT_LEFT<FRACTIONAL>(m1), low));
#else
    Datatype r1 = getRandomVal(PSELF);
#if RESHARE_BAKE_ACTIVE  // gated: consumes an extra PRNG draw
    if (bake_index >= 0)
        r1 = construct_mwk_r1_baked(r1, getRandomVal(PSELF), bake_index, SUB);
#endif
    return r1;
#endif
}

// Untruncated mask l = -r1 (TRUNC_DELAYED): fully bakeable, no image constraint.
template <typename Datatype, typename func_sub>
inline Datatype mwk_choose_r1_no_trunc(int bake_index, func_sub SUB)
{
#if A2B_CONV_BAKE_ACTIVE
    // A2B bake: prescribe P1's conv/FC triple share r1 = -lz1 (committed), so its output mask
    // l = -r1 = lz1 and [c] = bool(-(lz0+lz1)) matches. No getRandomVal draw (mask is derived), and
    // identical in PRE and LIVE. bake_index = layer-local output index (indexed g_a2b_lz access).
    return SUB(SET_ALL_ZERO(), a2b_bake_conv_mask<Datatype>((uint64_t)(bake_index < 0 ? 0 : bake_index), SUB));
#else
    Datatype r1 = getRandomVal(PSELF);
    if (bake_index >= 0)
    {
        Datatype l_t = r1;
        bake_reshare_mask(l_t, bake_index, SUB);  // no-op unless RESHARE_BAKE_ACTIVE && PARTY == 1
        r1 = SUB(SET_ALL_ZERO(), l_t);
    }
    return r1;
#endif
}
// P1-side live count of the SIM=1 reshare condition (see the counter comment above); the unit test prints it.
template <typename Datatype>
inline void reshare_sim_check(Datatype l, Datatype mask)
{
#if PARTY == 1 && DATTYPE == BITLENGTH
    if (current_phase != PHASE_LIVE)
        return;
    g_rb_checks++;
    g_rb_mismatch += (UINT_TYPE) l != (UINT_TYPE) mask;
#else
    (void) l; (void) mask;
#endif
}

std::vector<uint64_t> num_arithmetic_triples;
std::vector<uint64_t> num_ab2_arithmetic_triples;
std::vector<uint64_t> num_boolean_triples;
uint64_t num_boolean_addition_triples;
uint64_t num_multiplexer_triples;
uint64_t num_cot_triples;
std::vector<uint64_t> num_ab2_boolean_triples;
std::vector<uint64_t> triple_type_index;
std::vector<uint8_t*> triple_type;
/* uint64_t boolean_triple_index = 0; */
/* uint64_t num_arithmetic_triples = 0; */
/* uint64_t num_boolean_triples = 0; */
/* uint64_t triple_type_index = 0; */
/* uint8_t* triple_type; */

std::vector<uint64_t> total_num_boolean_output_triples;
std::vector<uint64_t> total_num_arithmetic_output_triples;


uint64_t total_arithmetic_triples_num = 0;
uint64_t total_boolean_triples_num = 0;
uint64_t total_arithmetic_triples_index = 0;
uint64_t total_boolean_triples_index = 0;

uint64_t arithmetic_triple_index = 0;
uint64_t boolean_triple_index = 0;
uint64_t curr_arithmetic_triple_index = 0;
uint64_t curr_boolean_triple_index = 0;
DATATYPE* arithmetic_triple_a = nullptr;
DATATYPE* arithmetic_triple_b = nullptr;
DATATYPE* arithmetic_triple_c = nullptr;
DATATYPE* boolean_triple_a = nullptr;
DATATYPE* boolean_triple_b= nullptr;
DATATYPE* boolean_triple_c = nullptr;

uint64_t total_ab2_arithmetic_triples_num = 0;
uint64_t total_ab2_boolean_triples_num = 0;
uint64_t total_ab2_arithmetic_triples_index = 0;
uint64_t total_ab2_boolean_triples_index = 0;

uint64_t curr_arithmetic_ab2_triple_index = 0;
uint64_t curr_boolean_ab2_triple_index = 0;
uint64_t arithmetic_ab2_triple_index = 0;
uint64_t boolean_ab2_triple_index = 0;
DATATYPE* arithmetic_ab2_triple_a = nullptr;
DATATYPE* arithmetic_ab2_triple_b = nullptr;
DATATYPE* arithmetic_ab2_triple_c = nullptr;
DATATYPE* boolean_ab2_triple_a = nullptr;
DATATYPE* boolean_ab2_triple_b = nullptr;
DATATYPE* boolean_ab2_triple_c = nullptr;


uint64_t curr_boolean_addition_triple_index = 0;
uint64_t boolean_addition_triple_index = 0;
DATATYPE* boolean_addition_triple_a = nullptr;
DATATYPE* boolean_addition_triple_b= nullptr;
DATATYPE* boolean_addition_triple_c = nullptr;

#if A2B_CONV_BAKE_ACTIVE
// Choose ia, derive lz = -untranspose(ia), and load the boolean-addition input buffers - ONCE, before
// either FUNCTION pass. getRandomVal(PSELF) is saved/restored so the function's own PSELF stream stays
// PRE<->LIVE synced. The caller then runs the boolean addition (generate_beaver_triples "BOOLEANADDITION")
// and calls a2b_bake_store_c() to capture this party's [c] = bool(-lz) share.
template <typename Datatype, typename func_sub>
inline void init_a2b_bake(uint64_t num_slices, func_sub SUB)
{
    constexpr int K = BITLENGTH;
    if (!g_a2b_lz.empty())
        return;  // generated once; both phases reuse it
    g_a2b_ia.assign(num_slices, SET_ALL_ZERO());
    g_a2b_lz.assign(num_slices, SET_ALL_ZERO());
    g_a2b_c.assign(num_slices, SET_ALL_ZERO());
    (void) SUB;  // the masks are derived value by value (UINT_TYPE arithmetic), independent of the lane layout
    // One group of K words: ia drawn, lz derived.
    auto group = [&](uint64_t base, auto draw)
    {
        Datatype ia[K];
        for (int i = 0; i < K; i++) { ia[i] = draw(base + i); g_a2b_ia[base + i] = ia[i]; }
        // The A2B slices a group of K words as orthogonalize_boolean(unorthogonalize_arithmetic(x)) (prepare_A2B_S1/S2).
        // So -lz, value by value (DATTYPE values: K words x DATTYPE / K lanes), is unorthogonalize_boolean(ia), and lz
        // is packed back with orthogonalize_arithmetic: then ortho(-lz) == ia. With DATTYPE == BITLENGTH the arithmetic
        // packing is the identity and real_ortho is self-inverse (the original single-batch construction).
        alignas(sizeof(Datatype)) UINT_TYPE t2[DATTYPE];
        alignas(sizeof(Datatype)) UINT_TYPE m[DATTYPE];
        Datatype tmp[K];
        for (int i = 0; i < K; i++) tmp[i] = ia[i];
        unorthogonalize_boolean(tmp, t2);
        for (int j = 0; j < DATTYPE; j++)
        {
            m[j] = (UINT_TYPE) 0 - t2[j];  // lz, numeric
#if A2B_P1_IMAGE_MASKS
            // TD=0: P1's conv/FC output mask is l1 = TRUNC(-r1), and TRUNC (FUNC_TRUNC = OP_TRUNC under
            // SKIP_PRE=0) is a LOGICAL shift - its image has the top FRACTIONAL bits ZERO. Constrain the
            // committed m1 the same way (zero the top F bits; NOT sign-extension) and RE-derive
            // ia = bool(-m1) so the early boolean addition still yields [c] = bool(-(lz0+m1)). The remask
            // path (no trunc) also uses this m1 - a validly-masked, just constrained, value - stays correct.
            m[j] &= (((UINT_TYPE) 1 << (BITLENGTH - FRACTIONAL)) - (UINT_TYPE) 1);
            t2[j] = (UINT_TYPE) 0 - m[j];
#endif
        }
#if A2B_P1_IMAGE_MASKS
        Datatype ia_new[K];
        orthogonalize_boolean(t2, ia_new);  // ia = bool(-lz) = bool(-m1)
        for (int i = 0; i < K; i++) g_a2b_ia[base + i] = ia_new[i];
#endif
        Datatype lz[K];
        orthogonalize_arithmetic(m, lz);
        for (int i = 0; i < K; i++) g_a2b_lz[base + i] = lz[i];
    };
#if RANDOM_ALGORITHM == 2 && USE_SSL_AES == 0
    // Counter-mode values under this party's key with a tweak of their own (prf_value): independent of the generator
    // stream the passes draw their own masks from (a replay of that stream would make lz a known function of other
    // masks of the same party, whose masked values are public too), and random access, so the groups are derived on
    // several threads (this runs in the OT phase).
    prf_value_check();
    const uint64_t groups = num_slices / K;
    const uint64_t T = std::max<uint64_t>(1, std::min<uint64_t>(A2B_BAKE_INIT_THREADS, groups));
    std::vector<std::thread> workers;
    for (uint64_t t = 0; t < T; t++)
        workers.emplace_back([&, t] {
            for (uint64_t g = groups * t / T; g < groups * (t + 1) / T; g++)
                group(g * K, [](uint64_t slot) { return prf_value(kTweakA2bIa, slot); });
        });
    for (auto& w : workers) w.join();
#else
    for (uint64_t base = 0; base + K <= num_slices; base += K)  // (as before the PRF path)
        group(base, [](uint64_t) { return getRandomVal(PSELF); });
#endif
#if A2B_RESIDUAL_COMMIT && PARTY == 1
    // Residual sums with a committed other addend (a2b_residual_committed): lz_1 = m_a + m_b, with m_a the image
    // value drawn above (the partner's mask) and m_b the other addend's committed mask; ia = bool(-lz_1) again.
    for (size_t k = 0; k < g_residual_sums.size(); k++)
    {
        const ResidualSum& r = g_residual_sums[k];
        if (!a2b_residual_committed((int) k) || r.slots == 0)
            continue;
        if (r.slot_base % K != 0 || r.slot_base + r.slots > num_slices)
        {
            fprintf(stderr, "A2B_CONV_BAKE: residual sum %zu has slots %lu..%lu of %lu\n", k,
                    (unsigned long) r.slot_base, (unsigned long) (r.slot_base + r.slots), (unsigned long) num_slices);
            std::abort();
        }
        for (uint64_t base = r.slot_base; base < r.slot_base + r.slots; base += K)
        {
            alignas(sizeof(Datatype)) UINT_TYPE t2[DATTYPE];  // DATTYPE == BITLENGTH: one value per word
            for (int j = 0; j < K; j++)
            {
                const UINT_TYPE lz1 = (UINT_TYPE) g_a2b_lz[base + j] +
                                      (UINT_TYPE) a2b_residual_other_mask((int) k, base + j - r.slot_base);
                g_a2b_lz[base + j] = (DATATYPE) lz1;
                t2[j] = (UINT_TYPE) 0 - lz1;
            }
            Datatype ia[K];
            orthogonalize_boolean(t2, ia);
            for (int i = 0; i < K; i++) g_a2b_ia[base + i] = ia[i];
        }
    }
#endif
    // Hand this party's ia to the boolean-addition input buffer (P0 -> a, P1 -> b), in slice order.
    for (uint64_t e = 0; e < num_slices; e++)
#if PARTY == 0
        boolean_addition_triple_a[e] = g_a2b_ia[e];
#else
        boolean_addition_triple_b[e] = g_a2b_ia[e];
#endif
#if PARTY == 0
    a2b_shift_inputs(boolean_addition_triple_a);
#else
    a2b_shift_inputs(boolean_addition_triple_b);
#endif
}

// After the early boolean addition has produced boolean_addition_triple_c, capture this party's [c] share.
inline void a2b_bake_store_c(uint64_t num_slices)
{
    for (uint64_t e = 0; e < num_slices && e < g_a2b_c.size(); e++)
        g_a2b_c[e] = boolean_addition_triple_c[e];
}

#if A2B_MASK_PASS_ACTIVE
// Set by the executer, run by the mask-only forward once it is done: the Boolean addition on the recorded masks
inline std::function<void()> g_mask_pass_hook;

// UC3: the mask-only forward's ReLU also records its input masks at its A2B slots (padding: 0), which advance as
// get_msb_range advances them (BITLENGTH per packed sint), in step with g_relu_base.
#if TS1_FUSED_ACTIVE
// TS1: the first ReLU's inputs still carry the data owner's input sharing (one party's mask share is 0 or the value
// itself), which the shifted design's rounding does not suit (it is centred for two random shares). The mask-only
// forward leaves their slots' committed masks (init_a2b_bake), and get_msb_range moves the inputs onto them (the
// rebase message, as without the mask pass) while RELU sets this flag (in every pass):
inline bool g_a2b_rebase_now = false;
#endif
template <typename Datatype, typename Share, typename A>  // A: Additive_Share<Datatype, Share>
void a2b_mask_pass_record(const A* in, int len, bool first_relu = false)
{
    if constexpr (requires(const Share& s) { s.get_mask(); })
    {
        const uint64_t base = g_a2b_layer_base;
        const uint64_t slots = (uint64_t) ((len + BITLENGTH - 1) / BITLENGTH) * BITLENGTH;
        if (base + slots > g_a2b_lz.size())
            mask_pass_abort("more ReLU inputs than A2B slots");
#if TS1_FUSED_ACTIVE
        if (first_relu)
        {
            g_a2b_layer_base = base + slots;  // the committed masks stay
            return;
        }
#else
        (void) first_relu;
#endif
        for (int v = 0; v < len; v++) g_a2b_lz[base + v] = in[v].get_mask();
        for (uint64_t v = len; v < slots; v++) g_a2b_lz[base + v] = SET_ALL_ZERO();
        g_a2b_layer_base = base + slots;
    }
    else
        mask_pass_abort("not a preprocessing share");
}

// After the mask-only forward: the Boolean addition's input of every slot is bool(-lambda_v) of the recorded mask
// (the construction of init_a2b_bake, from lz instead of towards it); [c] then fits the actual masks.
inline void a2b_mask_pass_commit(uint64_t num_slices)
{
    constexpr int K = BITLENGTH;
    for (uint64_t base = 0; base + K <= num_slices; base += K)
    {
        alignas(sizeof(DATATYPE)) UINT_TYPE t2[DATTYPE];
        for (int j = 0; j < K; j++) t2[j] = (UINT_TYPE) 0 - (UINT_TYPE) g_a2b_lz[base + j];
        DATATYPE ia[K];
        orthogonalize_boolean(t2, ia);
        for (int i = 0; i < K; i++)
        {
            g_a2b_ia[base + i] = ia[i];
#if PARTY == 0
            boolean_addition_triple_a[base + i] = ia[i];
#else
            boolean_addition_triple_b[base + i] = ia[i];
#endif
        }
    }
#if PARTY == 0
    a2b_shift_inputs(boolean_addition_triple_a);
#else
    a2b_shift_inputs(boolean_addition_triple_b);
#endif
}
#endif
#endif

#if TS1_FUSED_ACTIVE || A2B_DCUT_ACTIVE
// The A2B input transform of the running ReLU (get_msb_range, after the rebase, online only: the other passes have
// no m). The A2B with the bake reads only the inputs' m and [c]; see a2b_xform_input.
enum class A2bXform
{
    None,
    Ts1Full,   // TS1, full design with the cut: m := M (offset 2^(l-2)) - kTs1A2bLow, [c] shifted (g_a2b_c_shift)
    Ts1Shift,  // TS1, shifted design: m := the shifted value's public part M0 (ts1_shift_m0)
    DCut       // A2B_DCUT_ACTIVE: m := m >> F, restored after the A2B
};
inline A2bXform g_a2b_xform = A2bXform::None;
inline DATATYPE* g_a2b_xform_m = nullptr;   // by value: Ts1Full the truncated value's public part M, DCut the saved m
inline DATATYPE* g_a2b_xform_sk = nullptr;  // by value: Ts1Full sK
inline UINT_TYPE g_a2b_xform_factor = 1;    // Ts1Shift: the public factor
#endif

#if TS1_FUSED_ACTIVE
// TS1 fused into the ReLU: a delayed ReLU input z = m - lambda = m + nu (scale 2^(2F), nu = -lambda = nu_0 + nu_1,
// nu_i = -lz_i) is truncated without a message of its own, from public functions of m and preprocessed functions of
// nu, which come with the A2B bake's Boolean addition. Two designs:
//
// Shifted (TS1_CUT_ACTIVE without TS1_LOW_CARRY, and every ReLU with a fused pooling factor): the Boolean addition adds
// the shifted shares rho_i = (f nu_i) >> F (a2b_shift_inputs, f a public factor, 1 or 1/denom with t' bits), over
// the low l' = l - F bits (the cut adder: 26 instead of 31 rounds). Then (f z) >> F = (f m >> F) + rho + E mod 2^l',
// E in {0, 1, 2} the dropped low carries; the ReLU's A2B converts y1 = M0 + rho, M0 = ((f m + 2^(F-1)) >> F) + 1
// (rounded and centred for two random mask shares: unbiased; see the first ReLU in a2b_mask_pass_record), with
// the cut, and TS1 in the small ring lifts y1 >> t' to l bits: on u = -y1 with c = -M0 + 2^(l'-2) (1-bit slack),
// r = rho, K = 2^(l'-1-t'), w the carry into bit l'-1 and r_msb bit l'-1 of rho_0 + rho_1:
//   y = M - (la + s K r_msb),  M = (2^(l'-2) >> t') - c' - K MSB(c),  s = 1 - 2 MSB(c),  c' = (c mod 2^(l'-1)) >> t',
//   la = -((rho_0 mod 2^(l'-1)) >> t') - ((rho_1 mod 2^(l'-1)) >> t') + K w [- w_t].
// With t' = 0, y = y1 exactly (the A2B and the bit injection see the same value), off by 1 - E in {-1, 0, 1}. With
// t' > 0 la also takes w_t, the carry into bit t' of rho_0 + rho_1 (a COT of l - 1 bits, for these ReLUs only): the
// lift is then exact TS1, y = floor(y1 / 2^t') + e0, e0 in {0, 1} with mean frac(y1 / 2^t') (unbiased, and >= 0 where
// DReLU(y1) = 1). Leaving w_t out costs half an LSB on average, which the small averages of a pooling do not survive.
//
// Full (TS1_LOW_CARRY, or no cut): TS1 on u = -z in the full ring, r = nu: with a_i = nu_i mod 2^(l-1), K = 2^(l-1-F),
// w, w_t the carries into bits l-1 and F of nu_0 + nu_1 (the Boolean addition runs full width):
//   y = M - (la + s K r_msb),  M = (offset >> F) - c' - K MSB(c),  c = -m + offset,  c' = (c >> F) mod 2^(l-1-F),
//   la = -(a_0 >> F) - (a_1 >> F) + K w [- w_t].
// offset 2^(l-1) when DReLU is taken of z (no cut: exact for z >= 0, the others are multiplied by 0), 2^(l-2) with the
// cut (DReLU of y: its A2B takes the bake's [c] shifted by F slices and M - kTs1A2bLow). Off by 0 / +1 (TS1_LOW_CARRY)
// or -1 / 0 / +1.
//
// M and s are public, la and r_msb (arithmetic, correct mod 2^(F+1+t'): only used times K) preprocessed shares; the
// bit injection takes y with the products [lambda_b la] (its usual one) and [lambda_b r_msb] (ts1_generate_products).
// Indexed by a compact index over the TS1 ReLUs' A2B slots (Datatype words, in forward order).
struct Ts1Range
{
    uint64_t slot_base, slots, compact_base;
    bool shift;        // the shifted design
    UINT_TYPE factor;  // shifted design: the public factor (1: none)
    int tp;            // shifted design: the lift's truncation t'
};
inline std::vector<Ts1Range> g_ts1_ranges;  // recorded by the INIT pass
inline uint64_t g_ts1_slots = 0;
inline std::vector<DATATYPE> g_ts1_la;      // this party's share of la
inline std::vector<DATATYPE> g_ts1_r;       // ... of r_msb
inline std::vector<DATATYPE> g_ts1_mux_b;   // the bit injection's Boolean mask, per group (preprocessing pass)
inline std::vector<DATATYPE> g_ts1_mux_c;   // ... of lambda_b * r_msb
constexpr UINT_TYPE kTs1K = (UINT_TYPE) 1 << (BITLENGTH - 1 - FRACTIONAL);
constexpr int kTs1Lp = BITLENGTH - FRACTIONAL;  // l'
#if TS1_CUT_ACTIVE
constexpr UINT_TYPE kTs1A2bLow = TS1_LOW_CARRY == 1 ? 0 : 1;
#endif

// Full design: c = -m + offset, c' = (c >> F) mod 2^(l-1-F), M = (offset >> F) - c' - K MSB(c), sK = (1 - 2 MSB(c)) K
inline void ts1_public(DATATYPE m, UINT_TYPE offset, DATATYPE& M, DATATYPE& sk)
{
    const DATATYPE K = PROMOTE(kTs1K);
    const DATATYPE c = OP_ADD(OP_SUB(SET_ALL_ZERO(), m), PROMOTE(offset));
    const DATATYPE cm = OP_SHIFT_LOG_RIGHT<BITLENGTH - 1>(c);
    const DATATYPE cp = FUNC_AND(OP_SHIFT_LOG_RIGHTF(c, FRACTIONAL), PROMOTE(kTs1K - 1));
    sk = OP_MULT(OP_SUB(PROMOTE(1), OP_ADD(cm, cm)), K);
    M = OP_SUB(OP_SUB(PROMOTE(offset >> FRACTIONAL), cp), OP_MULT(cm, K));
}

// Shifted design: the A2B's public part M0 = ((factor m + 2^(F-1)) >> F) + 1 mod 2^l' (a2b_xform_input) ...
inline DATATYPE ts1_shift_m0(DATATYPE m, UINT_TYPE factor)
{
    const DATATYPE mf = OP_ADD(OP_MULT(m, PROMOTE(factor)), PROMOTE((UINT_TYPE) 1 << (FRACTIONAL - 1)));  // rounded
    return FUNC_AND(OP_ADD(OP_SHIFT_LOG_RIGHT<FRACTIONAL>(mf), PROMOTE(1)), PROMOTE((((UINT_TYPE) 1 << kTs1Lp) - 1)));
}
// ... and the lift's, from M0 (the bit injection): c = -M0 + 2^(l'-2) mod 2^l', c' = (c mod 2^(l'-1)) >> t',
// M = (2^(l'-2) >> t') - c' - K MSB(c), sK = (1 - 2 MSB(c)) K, K = 2^(l'-1-t')
inline void ts1_lift_shift(DATATYPE M0, int tp, DATATYPE& M, DATATYPE& sk)
{
    constexpr int Lp = kTs1Lp;
    const UINT_TYPE K = (UINT_TYPE) 1 << (Lp - 1 - tp);
    const UINT_TYPE off = (UINT_TYPE) 1 << (Lp - 2);
    const DATATYPE c = FUNC_AND(OP_ADD(OP_SUB(SET_ALL_ZERO(), M0), PROMOTE(off)), PROMOTE((((UINT_TYPE) 1 << Lp) - 1)));
    const DATATYPE cm = OP_SHIFT_LOG_RIGHT<Lp - 1>(c);
    const DATATYPE cp = OP_SHIFT_LOG_RIGHTF(FUNC_AND(c, PROMOTE((off << 1) - 1)), tp);
    sk = OP_MULT(OP_SUB(PROMOTE(1), OP_ADD(cm, cm)), PROMOTE(K));
    M = OP_SUB(OP_SUB(PROMOTE(off >> tp), cp), OP_MULT(cm, PROMOTE(K)));
}

inline void ts1_record_range(uint64_t slot_base, uint64_t slots, bool shift, UINT_TYPE factor, int tp)
{
    g_ts1_ranges.push_back({slot_base, slots, g_ts1_slots, shift, factor, tp});
    g_ts1_slots += slots;
}

// The TS1 ReLU whose A2B slots start at slot_base (PRE and LIVE pass)
inline const Ts1Range& ts1_range(uint64_t slot_base, uint64_t slots)
{
    auto it = std::lower_bound(g_ts1_ranges.begin(), g_ts1_ranges.end(), slot_base,
                               [](const Ts1Range& r, uint64_t s) { return r.slot_base < s; });
    if (it == g_ts1_ranges.end() || it->slot_base != slot_base || it->slots != slots)
    {
        fprintf(stderr, "TS1: no tuples for a ReLU at A2B slot %lu (%lu slots)\n", (unsigned long) slot_base,
                (unsigned long) slots);
        std::abort();
    }
    return *it;
}

// The OTs ts1_generate_tuples and ts1_generate_products take (Iface::ot_demand_hint)
inline uint64_t ts1_ot_demand()
{
    return g_ts1_slots * DATTYPE / BITLENGTH * (2 + TS1_LOW_CARRY + 2);
}

// The TS1 groups (BITLENGTH compact slots each) by width base + t' (t' = 0 for the full design), ascending
inline std::map<int, std::vector<uint64_t>> ts1_group_widths(int base)
{
    std::map<int, std::vector<uint64_t>> w;
    for (const Ts1Range& r : g_ts1_ranges)
    {
        auto& gs = w[base + (r.shift ? r.tp : 0)];
        for (uint64_t j = 0; j < r.slots / BITLENGTH; j++) gs.push_back(r.compact_base / BITLENGTH + j);
    }
    return w;
}

// After the bake's Boolean addition (a2b_bake_store_c): this party's Boolean shares of the carries are [c] ^ (its own
// input to the addition), and w, r_msb (and w_t) become arithmetic shares with one narrow COT each.
inline void ts1_generate_tuples(const std::string& ip, int port)
{
    constexpr int K = BITLENGTH;
    constexpr int F = FRACTIONAL;
    constexpr int Lp = kTs1Lp;
    bool any_full = false;
    for (const Ts1Range& r : g_ts1_ranges) any_full |= !r.shift;
    if (any_full && a2b_adder_lo(BITLENGTH) != 0)
    {
        fprintf(stderr, "TS1: the Boolean addition stops below bit %d, the full design needs its carries up to the top\n",
                K - 1);
        std::abort();
    }
    const uint64_t groups = g_ts1_slots / K;
    const uint64_t n = groups * DATTYPE;  // values
    g_ts1_la.assign(g_ts1_slots, SET_ALL_ZERO());
    g_ts1_r.assign(g_ts1_slots, SET_ALL_ZERO());
    g_ts1_mux_b.assign(groups, SET_ALL_ZERO());
    g_ts1_mux_c.assign(g_ts1_slots, SET_ALL_ZERO());
    if (n == 0)
        return;
    std::vector<const Ts1Range*> grp_range(groups);
    std::vector<uint64_t> grp_slot(groups);
    for (const Ts1Range& r : g_ts1_ranges)
        for (uint64_t j = 0; j < r.slots / K; j++)
            grp_range[r.compact_base / K + j] = &r, grp_slot[r.compact_base / K + j] = r.slot_base + j * K;
    std::vector<UINT_TYPE> a_hi(n);  // this party's (a_i >> F) (full) or ((rho_i mod 2^(l'-1)) >> t') (shifted)
    std::vector<uint8_t> u[3];       // this party's share bits of w, r_msb, w_t
    for (auto& b : u) b.assign(n, 0);
    {
        const uint64_t T = std::max<uint64_t>(1, std::min<uint64_t>(A2B_BAKE_INIT_THREADS, groups));
        std::vector<std::thread> workers;
        for (uint64_t t = 0; t < T; t++)
            workers.emplace_back([&, t] {
                for (uint64_t g = groups * t / T; g < groups * (t + 1) / T; g++)
                {
                    const Ts1Range& r = *grp_range[g];
                    const uint64_t slot = grp_slot[g];
                    alignas(sizeof(DATATYPE)) UINT_TYPE nu[DATTYPE], cs[DATTYPE];  // nu_i and this party's share of the sum
                    a2b_group_values(&g_a2b_ia[slot], nu);
                    a2b_group_values(&g_a2b_c[slot], cs);
                    for (int j = 0; j < DATTYPE; j++)
                    {
                        const uint64_t v = g * DATTYPE + j;
                        if (r.shift)
                        {
                            const UINT_TYPE x = a2b_shifted(nu[j], r.factor);  // the addition's input rho_i
                            const UINT_TYPE carry = cs[j] ^ x;
                            a_hi[v] = (x & (((UINT_TYPE) 1 << (Lp - 1)) - 1)) >> r.tp;
                            u[0][v] = (carry >> (Lp - 1)) & 1;
                            u[1][v] = (cs[j] >> (Lp - 1)) & 1;
                            u[2][v] = r.tp > 0 ? (carry >> r.tp) & 1 : 0;
                        }
                        else
                        {
                            const UINT_TYPE carry = cs[j] ^ nu[j];
                            a_hi[v] = (nu[j] & (((UINT_TYPE) 1 << (K - 1)) - 1)) >> F;
                            u[0][v] = (carry >> (K - 1)) & 1;
                            u[1][v] = (cs[j] >> (K - 1)) & 1;
                            u[2][v] = (carry >> F) & 1;
                        }
                    }
                }
            });
        for (auto& w : workers) w.join();
    }
    // arithmetic share of u_0 ^ u_1 = u_0 + u_1 - 2 u_0 u_1, the product by a COT of `width` bits (P0 the correlation
    // u_0, P1 the choice u_1): correct mod 2^(width + 1). Into out (n values) at the values `at` (all if empty).
    auto b2a = [&](const std::vector<uint8_t>& bit, int width, std::vector<UINT_TYPE>& out,
                   const std::vector<uint64_t>& at = {}) {
        const uint64_t cnt = at.empty() ? n : at.size();
        if (cnt == 0)
            return;
        auto idx = [&](uint64_t k) { return at.empty() ? k : at[k]; };
        std::vector<UINT_TYPE> c(cnt);
#if PARTY == 0
        std::vector<UINT_TYPE> corr(cnt);
        for (uint64_t k = 0; k < cnt; k++) corr[k] = bit[idx(k)];
        Iface::generateCOT(CHEETAH_PARTY, corr.data(), nullptr, c.data(), (unsigned) cnt, ip, port + CHEETAH_PORT_OFFSET,
                           CHEETAH_THREADS, CHEETAH_IO_OFFSET, width);
#else
        std::vector<uint8_t> packed((cnt + 7) / 8, 0);
        for (uint64_t k = 0; k < cnt; k++) packed[k / 8] |= (uint8_t) (bit[idx(k)] << (k % 8));
        Iface::generateCOT(CHEETAH_PARTY, nullptr, packed.data(), c.data(), (unsigned) cnt, ip, port + CHEETAH_PORT_OFFSET,
                           CHEETAH_THREADS, CHEETAH_IO_OFFSET, width);
#endif
        for (uint64_t k = 0; k < cnt; k++) out[idx(k)] = (UINT_TYPE) bit[idx(k)] - (UINT_TYPE) 2 * c[k];
    };
    std::vector<UINT_TYPE> aw(n, 0), am(n, 0), at(n, 0);
    // K w and K r_msb: mod 2^(F+1+t') is enough, one COT width per t' (only a pooling's ReLUs have t' > 0)
    const auto widths = ts1_group_widths(FRACTIONAL);
    for (const auto& [width, gs] : widths)
    {
        std::vector<uint64_t> vs;  // empty: all values
        if (widths.size() > 1)
            for (uint64_t g : gs)
                for (int j = 0; j < DATTYPE; j++) vs.push_back(g * DATTYPE + j);
        b2a(u[0], width, aw, vs);
        b2a(u[1], width, am, vs);
    }
    std::vector<uint64_t> low;  // the values that take w_t: full design with TS1_LOW_CARRY, shifted design with t' > 0
    for (uint64_t g = 0; g < groups; g++)
        if ((grp_range[g]->shift && grp_range[g]->tp > 0) || (!grp_range[g]->shift && TS1_LOW_CARRY == 1))
            for (int j = 0; j < DATTYPE; j++) low.push_back(g * DATTYPE + j);
    if (!low.empty())
        b2a(u[2], K - 1, at, low);
    for (uint64_t g = 0; g < groups; g++)
    {
        const Ts1Range& r = *grp_range[g];
        const UINT_TYPE Kg = r.shift ? (UINT_TYPE) 1 << (Lp - 1 - r.tp) : kTs1K;
        alignas(sizeof(DATATYPE)) UINT_TYPE la[DATTYPE], rr[DATTYPE];
        for (int j = 0; j < DATTYPE; j++)
        {
            const uint64_t v = g * DATTYPE + j;
            la[j] = (UINT_TYPE) 0 - a_hi[v] + Kg * aw[v] - at[v];  // at: 0 where w_t is not taken
            rr[j] = am[v];
        }
        orthogonalize_arithmetic(la, &g_ts1_la[g * K]);
        orthogonalize_arithmetic(rr, &g_ts1_r[g * K]);
    }
}

// After the preprocessing pass recorded the bit injections' Boolean masks: [lambda_b r_msb] by a narrow multiplexer
// (mod 2^(F+1+t'), one call per t')
inline void ts1_generate_products(const std::string& ip, int port)
{
    constexpr int K = BITLENGTH;
    const uint64_t groups = g_ts1_slots / K;
    if (groups == 0)
        return;
    const auto widths = ts1_group_widths(FRACTIONAL + 1);
    for (const auto& [width, gs] : widths)
    {
        if (widths.size() == 1)
        {
            generateMultiplexerDummyTriples(g_ts1_r.data(), g_ts1_mux_b.data(), g_ts1_mux_c.data(), BITLENGTH,
                                            groups * DATTYPE, ip, port, width);
            break;
        }
        std::vector<DATATYPE> r(gs.size() * K), b(gs.size()), c(gs.size() * K);
        for (uint64_t k = 0; k < gs.size(); k++)
        {
            std::copy_n(&g_ts1_r[gs[k] * K], K, &r[k * K]);
            b[k] = g_ts1_mux_b[gs[k]];
        }
        generateMultiplexerDummyTriples(r.data(), b.data(), c.data(), BITLENGTH, gs.size() * DATTYPE, ip, port, width);
        for (uint64_t k = 0; k < gs.size(); k++) std::copy_n(&c[k * K], K, &g_ts1_mux_c[gs[k] * K]);
    }
}
#endif

#if MASK_FORWARD_ACTIVE

// The masks of the truncations outside ReLUs (pooling, delayed conv truncations, the data owner's first layer): a
// stream that restarts with every forward.
inline uint64_t g_lin_counter = 0;

// The committed ReLU output masks, before the preprocessing pass (g_relu_slots from the INIT pass)
inline void init_relu_out_masks()
{
    prf_value_check();
    g_relu_out.resize(g_relu_slots);
    for (uint64_t k = 0; k < g_relu_slots; k++) g_relu_out[k] = prf_value(kTweakRelu, k);
}

// The mask-only forward's ReLU: its outputs take the committed masks of its slots
template <typename Datatype, typename Share, typename A>  // A: Additive_Share<Datatype, Share>
void mask_pass_relu_outputs(int len, A* out, uint64_t base)
{
    if constexpr (requires { Share(Datatype{}); })
    {
        if (base + (uint64_t) len > g_relu_out.size())
            mask_pass_abort("more ReLU outputs than committed masks");
        for (int v = 0; v < len; v++) out[v] = A(Share(g_relu_out[base + v]));
    }
    else
        mask_pass_abort("not a preprocessing share");
}
#endif

// The fresh mask of a truncation's output (see prf_value)
template <typename Datatype>
inline Datatype lin_mask()
{
#if MASK_FORWARD_ACTIVE
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_stream || tl_pre)
        mask_pass_abort("a truncation inside a parallel level");
#endif
    return prf_value(kTweakTrunc, g_lin_counter++);
#else
    return getRandomVal(PSELF);
#endif
}

// The output mask of bit-injection output i: inside a ReLU with a mask-only forward the committed one of its slot,
// otherwise a fresh draw.
template <typename Datatype>
inline Datatype bi_output_mask(int i)
{
#if MASK_FORWARD_ACTIVE
    if (tl_bi_slot >= 0)
        return g_relu_out[(uint64_t) tl_bi_slot + (uint64_t) i];
#endif
    (void) i;
    return getRandomVal(PSELF);
}

uint64_t curr_multiplexer_triple_index = 0;
uint64_t arithmetic_multiplexer_triple_index = 0;
uint64_t boolean_multiplexer_triple_index = 0;
DATATYPE* multiplexer_triple_a = nullptr;
DATATYPE* multiplexer_triple_b= nullptr;
DATATYPE* multiplexer_triple_c = nullptr;

uint64_t curr_cot_triple_index = 0;
uint64_t arithmetic_cot_triple_index = 0;
uint64_t boolean_cot_triple_index = 0;
DATATYPE* cot_triple_a = nullptr;
DATATYPE* cot_triple_b= nullptr;
DATATYPE* cot_triple_c = nullptr;
        

DATATYPE** conv_triple_w = nullptr;
DATATYPE** conv_triple_x = nullptr;
DATATYPE* conv_triple_y = nullptr;
uint64_t curr_conv_triple_index = 0;
uint64_t num_conv_c_triples = 0;
std::vector<ConvolutionParameter> conv_triple_params;

// Writes of the preprocessing share into its append-only streams. Inside a parallel preprocessing level
// (stream_parallel_for in PHASE_PRE) they go through the worker's cursor (tl_pre) instead of the global index.
inline void put_triple_type(int r, uint8_t t)
{
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_pre)
    {
        *tl_pre->type[r]++ = t;
        return;
    }
#endif
    triple_type[r][triple_type_index[r]++] = t;
}
inline void put_boolean_addition_input(DATATYPE ia)  // P0 the a side, P1 the b side
{
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_pre)
    {
        *tl_pre->bool_add++ = ia;
        return;
    }
#endif
#if PARTY == 0
    boolean_addition_triple_a[boolean_addition_triple_index++] = ia;
#else
    boolean_addition_triple_b[boolean_addition_triple_index++] = ia;
#endif
}
inline void put_multiplexer_arith(DATATYPE v)
{
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_pre)
    {
        *tl_pre->mux_arith++ = v;
        return;
    }
#endif
    multiplexer_triple_a[arithmetic_multiplexer_triple_index++] = v;
}
inline void put_multiplexer_bool(DATATYPE v)
{
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_pre)
    {
        *tl_pre->mux_bool++ = v;
        return;
    }
#endif
    multiplexer_triple_b[boolean_multiplexer_triple_index++] = v;
}
inline void put_cot_arith(DATATYPE v)
{
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_pre)
    {
        *tl_pre->cot_arith++ = v;
        return;
    }
#endif
    cot_triple_a[arithmetic_cot_triple_index++] = v;
}

DATATYPE** fc_triple_w = nullptr;
DATATYPE** fc_triple_x = nullptr;
DATATYPE* fc_triple_y = nullptr;
uint64_t curr_fc_triple_index = 0;
uint64_t num_fc_c_triples = 0;
std::vector<FullyConnectedParameter> fc_triple_params;

DATATYPE** bc2D_triple_w = nullptr;
DATATYPE** bc2D_triple_x = nullptr;
DATATYPE* bc2D_triple_y = nullptr;
uint64_t curr_bc2D_triple_index = 0;
uint64_t num_bc2D_c_triples = 0;
std::vector<BatchNorm2DParameter> bc2D_triple_params;


 template <typename Datatype>
struct triple
{
    Datatype a;
    Datatype b;
    Datatype c;  // c = a*b
};

template <typename Datatype>
triple<Datatype> retrieveArithmeticTriple()
{
#if SKIP_PRE == 1
    return triple<Datatype>{SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()};
#else
    const uint64_t j = stream_index(IDX_ARITH, curr_arithmetic_triple_index)++;
    return triple<Datatype>{arithmetic_triple_a[j], arithmetic_triple_b[j], arithmetic_triple_c[j]};
#endif
}

template <typename Datatype>
triple<Datatype> retrieveBooleanTriple()
{
#if SKIP_PRE == 1
    return triple<Datatype>{SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()};
#else
    const uint64_t j = stream_index(IDX_BOOL, curr_boolean_triple_index)++;
    return triple<Datatype>{boolean_triple_a[j], boolean_triple_b[j], boolean_triple_c[j]};
    /* return triple<Datatype>{boolean_triple_a[boolean_triple_index], boolean_triple_b[boolean_triple_index],
     * boolean_triple_c[boolean_triple_index++]}; */
#endif
}

template <typename Datatype>
Beaver3Tuple<Datatype> retrieveBeaver3Tuple()
{
    if (!tl_stream)  // parallel levels start with the count at rest (see share_conversion.hpp)
        g_a2b_s1_pending = 0;  // an adder is consuming -> the prepare_A2B_S1 batch (if any) has ended
#if SKIP_PRE == 1
    return Beaver3Tuple<Datatype>{
        SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO(),
        SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()
    };
#else
    const uint64_t j_IDX_BEAVER3 = stream_index(IDX_BEAVER3, curr_beaver_3_triple_index);
    Beaver3Tuple<Datatype> tuple{
        beaver_3_tuples.a[j_IDX_BEAVER3],
        beaver_3_tuples.b[j_IDX_BEAVER3],
        beaver_3_tuples.c[j_IDX_BEAVER3],
        beaver_3_tuples.ab[j_IDX_BEAVER3],
        beaver_3_tuples.ac[j_IDX_BEAVER3],
        beaver_3_tuples.bc[j_IDX_BEAVER3],
        beaver_3_tuples.abc[j_IDX_BEAVER3]
    };
    stream_index(IDX_BEAVER3, curr_beaver_3_triple_index)++;
    return tuple;
#endif
}

template <typename Datatype>
Beaver4Tuple<Datatype> retrieveBeaver4Tuple()
{
#if SKIP_PRE == 1
    return Beaver4Tuple<Datatype>{
        SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO(),
        SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO(),
        SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO(),
        SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()
    };
#else
    const uint64_t j_IDX_BEAVER4 = stream_index(IDX_BEAVER4, curr_beaver_4_triple_index);
    Beaver4Tuple<Datatype> tuple{
        beaver_4_tuples.a[j_IDX_BEAVER4],
        beaver_4_tuples.b[j_IDX_BEAVER4],
        beaver_4_tuples.c[j_IDX_BEAVER4],
        beaver_4_tuples.d[j_IDX_BEAVER4],
        beaver_4_tuples.ab[j_IDX_BEAVER4],
        beaver_4_tuples.ac[j_IDX_BEAVER4],
        beaver_4_tuples.ad[j_IDX_BEAVER4],
        beaver_4_tuples.bc[j_IDX_BEAVER4],
        beaver_4_tuples.bd[j_IDX_BEAVER4],
        beaver_4_tuples.cd[j_IDX_BEAVER4],
        beaver_4_tuples.abc[j_IDX_BEAVER4],
        beaver_4_tuples.abd[j_IDX_BEAVER4],
        beaver_4_tuples.acd[j_IDX_BEAVER4],
        beaver_4_tuples.bcd[j_IDX_BEAVER4],
        beaver_4_tuples.abcd[j_IDX_BEAVER4]
    };
    stream_index(IDX_BEAVER4, curr_beaver_4_triple_index)++;
    return tuple;
#endif
}

template <typename Datatype>
RandomMultiplication<Datatype> retrieveRandomMultiplication()
{
#if SKIP_PRE == 1
    return RandomMultiplication<Datatype>{SET_ALL_ZERO(), SET_ALL_ZERO()};
#else
    const uint64_t j_IDX_RANDOM_MULT = stream_index(IDX_RANDOM_MULT, curr_random_multiplication_index);
    RandomMultiplication<Datatype> tuple{
        random_multiplication_a[j_IDX_RANDOM_MULT],
        random_multiplication_b[j_IDX_RANDOM_MULT]
    };
    stream_index(IDX_RANDOM_MULT, curr_random_multiplication_index)++;
    return tuple;
#endif
}

template <typename Datatype>
triple<Datatype> retrieveArithmeticAB2Triple()
{
#if SKIP_PRE == 1
    return triple<Datatype>{SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()};
#else
    const uint64_t j = stream_index(IDX_ARITH_AB2, curr_arithmetic_ab2_triple_index)++;
    return triple<Datatype>{arithmetic_ab2_triple_a[j], arithmetic_ab2_triple_b[j], arithmetic_ab2_triple_c[j]};
#endif
}

template <typename Datatype>
triple<Datatype> retrieveBooleanAB2Triple()
{
#if SKIP_PRE == 1
    return triple<Datatype>{SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()};
#else
    const uint64_t j = stream_index(IDX_BOOL_AB2, curr_boolean_ab2_triple_index)++;
    return triple<Datatype>{boolean_ab2_triple_a[j], boolean_ab2_triple_b[j], boolean_ab2_triple_c[j]};
#endif
}
    
    template <typename Datatype>
void storeArithmeticABTriple(const Datatype a, const Datatype b)
{
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_pre)
    {
        *tl_pre->ab_arith_a++ = a;
        *tl_pre->ab_arith_b++ = b;
        return;
    }
#endif
    arithmetic_triple_a[arithmetic_triple_index] = a;
    arithmetic_triple_b[arithmetic_triple_index] = b;
    arithmetic_triple_index++;
}

template <typename Datatype>
void storeBooleanABTriple(const Datatype a, const Datatype b)
{
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_pre)
    {
        *tl_pre->ab_bool_a++ = a;
        *tl_pre->ab_bool_b++ = b;
        return;
    }
#endif
    boolean_triple_a[boolean_triple_index] = a;
    boolean_triple_b[boolean_triple_index] = b; //B1 is not needed for the AB2 protocol
    boolean_triple_index++;
}

    template <typename Datatype>
void storeArithmeticAB2Triple(const Datatype a, const Datatype b)
{
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_pre)
    {
        *tl_pre->ab2_arith++ = PARTY == 0 ? a : b;
        return;
    }
#endif
#if PARTY == 0
    arithmetic_ab2_triple_a[arithmetic_ab2_triple_index] = a; //P0 holds A0 in plain in AB2 setting
#endif
#if PARTY != 0
    arithmetic_ab2_triple_b[arithmetic_ab2_triple_index] = b; //B1 is not needed for the AB2 protocol
#endif
    arithmetic_ab2_triple_index++;
}

template <typename Datatype>
void storeBooleanAB2Triple(const Datatype a, const Datatype b)
{
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_pre)
    {
        *tl_pre->ab2_bool++ = PARTY == 0 ? a : b;
        return;
    }
#endif
#if PARTY == 0
    boolean_ab2_triple_a[boolean_ab2_triple_index] = a;
#endif
#if PARTY != 0
    boolean_ab2_triple_b[boolean_ab2_triple_index] = b; //B1 is not needed for the AB2 protocol
#endif
    boolean_ab2_triple_index++;
}
    

template <typename Datatype>
Datatype retrieveBooleanLXLY()
{
#if SKIP_PRE == 1
    return SET_ALL_ZERO();
#else
    total_boolean_triples_index++;
    return boolean_triple_c[total_boolean_triples_index - 1];
#endif
}


template <typename Datatype>
Datatype retrieveArithmeticLXLY()
{
#if SKIP_PRE == 1
    return SET_ALL_ZERO();
#else
    total_arithmetic_triples_index++;
    return arithmetic_triple_c[total_arithmetic_triples_index - 1];
#endif
}


#if LX_TRIPLES == 1
void init_beaverAB(int rounds)
{
    arithmetic_triple_a = new DATATYPE[num_arithmetic_triples[rounds] ];
    arithmetic_triple_b = new DATATYPE[num_arithmetic_triples[rounds] ];
    boolean_triple_a = new DATATYPE[num_boolean_triples[rounds] ];
    boolean_triple_b = new DATATYPE[num_boolean_triples[rounds] ];
    // std::cout << "Initialized beaver AB for round " + std::to_string(rounds) + " with " + std::to_string(num_arithmetic_triples[rounds] * DATTYPE/BITLENGTH) + " arithmetic triples and " + std::to_string(num_boolean_triples[rounds] * DATTYPE) + " boolean triples.\n";
}

void init_beaverAB_arithmetic(int rounds)
{
    arithmetic_triple_a = new DATATYPE[num_arithmetic_triples[rounds] ];
    arithmetic_triple_b = new DATATYPE[num_arithmetic_triples[rounds] ];
}

void init_beaverAB_boolean(int rounds)
{
    boolean_triple_a = new DATATYPE[num_boolean_triples[rounds] ];
    boolean_triple_b = new DATATYPE[num_boolean_triples[rounds] ];
}


#if A2B_ONLINE_OPT == 1
void init_booleanAdditionBeaverAB()
{
    if(num_boolean_addition_triples == 0)
        return;
#if PARTY == 0
    boolean_addition_triple_a = new DATATYPE[num_boolean_addition_triples];
#else
    boolean_addition_triple_b = new DATATYPE[num_boolean_addition_triples];
#endif
}

void init_booleanAdditionBeaverC()
{
    boolean_addition_triple_c = new DATATYPE[num_boolean_addition_triples];
}

void deinit_booleanAdditionBeaverAB()
{
#if PARTY == 0
    delete[] boolean_addition_triple_a;
#else
    delete[] boolean_addition_triple_b;
#endif
}

void deinit_booleanAdditionBeaverC()
{
    delete[] boolean_addition_triple_c;
}
#endif

#if BIT_INJECTION_PREPROCESSING_OPT == 1

void init_multiplexerBeaverAB()
{
    multiplexer_triple_a = new DATATYPE[num_multiplexer_triples];
    multiplexer_triple_b = new DATATYPE[num_multiplexer_triples / BITLENGTH];
}

void init_multiplexerBeaverC()
{
    multiplexer_triple_c = new DATATYPE[num_multiplexer_triples];
}

void deinit_multiplexerBeaverAB()
{
    delete[] multiplexer_triple_a;
    delete[] multiplexer_triple_b;
}


void deinit_multiplexerBeaverC()
{
    delete[] multiplexer_triple_c;
}

#endif // BIT_INJECTION_PREPROCESSING_OPT == 1

#if BEAVER_N_TUPLES == 1

void init_beaver_3_tuples()
{
    beaver_3_tuples.a = new DATATYPE[num_beaver_3_tuples];
    beaver_3_tuples.b = new DATATYPE[num_beaver_3_tuples];
    beaver_3_tuples.c = new DATATYPE[num_beaver_3_tuples];
    beaver_3_tuples.ab = new DATATYPE[num_beaver_3_tuples];
    beaver_3_tuples.bc = new DATATYPE[num_beaver_3_tuples];
    beaver_3_tuples.ac = new DATATYPE[num_beaver_3_tuples];
    beaver_3_tuples.abc = new DATATYPE[num_beaver_3_tuples];
}

void init_beaver_4_tuples()
{
    beaver_4_tuples.a = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.b = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.c = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.d = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.ab = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.ac = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.ad = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.bc = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.bd = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.cd = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.abc = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.abd = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.acd = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.bcd = new DATATYPE[num_beaver_4_tuples];
    beaver_4_tuples.abcd = new DATATYPE[num_beaver_4_tuples];
}

void deinit_beaver_3_tuples()
{
    delete[] beaver_3_tuples.a;
    delete[] beaver_3_tuples.b;
    delete[] beaver_3_tuples.c;
    delete[] beaver_3_tuples.ab;
    delete[] beaver_3_tuples.ac;
    delete[] beaver_3_tuples.bc;
    delete[] beaver_3_tuples.abc;
}

void deinit_beaver_4_tuples()
{
    delete[] beaver_4_tuples.a;
    delete[] beaver_4_tuples.b;
    delete[] beaver_4_tuples.c;
    delete[] beaver_4_tuples.d;
    delete[] beaver_4_tuples.ab;
    delete[] beaver_4_tuples.ac;
    delete[] beaver_4_tuples.ad;
    delete[] beaver_4_tuples.bc;
    delete[] beaver_4_tuples.bd;
    delete[] beaver_4_tuples.cd;
    delete[] beaver_4_tuples.abc;
    delete[] beaver_4_tuples.abd;
    delete[] beaver_4_tuples.acd;
    delete[] beaver_4_tuples.bcd;
    delete[] beaver_4_tuples.abcd;
}

#endif // BEAVER_N_TUPLES == 1

void init_random_multiplications()
{
    random_multiplication_a = new DATATYPE[num_random_multiplications];
    random_multiplication_b = new DATATYPE[num_random_multiplications];
}

void deinit_random_multiplications()
{
    if (random_multiplication_a != nullptr) {
        delete[] random_multiplication_a;
        random_multiplication_a = nullptr;
    }
    if (random_multiplication_b != nullptr) {
        delete[] random_multiplication_b;
        random_multiplication_b = nullptr;
    }
}

#if BIT_INJECTION_PREPROCESSING_OPT == 1

void init_cotBeaverAB()
{
#if PARTY == 0
    cot_triple_a = new DATATYPE[num_cot_triples];
#else
    cot_triple_a = multiplexer_triple_b; //reuse lb share
#endif
}

void init_cotBeaverC()
{
    cot_triple_c = new DATATYPE[num_cot_triples];
}

void deinit_cotBeaverAB()
{
#if PARTY == 0
    delete[] cot_triple_a;
#else
    cot_triple_a = nullptr; // no need to delet since multiplexer is reused
#endif
}

void deinit_cotBeaverC()
{
    delete[] cot_triple_c;
}

#endif // BIT_INJECTION_PREPROCESSING_OPT == 1


void init_beaverC(int rounds)
{
    arithmetic_triple_c = new DATATYPE[num_arithmetic_triples[rounds] ];
    boolean_triple_c = new DATATYPE[num_boolean_triples[rounds] ];
    // std::cout << "Initialized beaver C for round " + std::to_string(rounds) + " with " + std::to_string(num_arithmetic_triples[rounds] * DATTYPE/BITLENGTH) + " arithmetic triples and " + std::to_string(num_boolean_triples[rounds] * DATTYPE) + " boolean triples.\n";
}

void init_beaverC_arithmetic(int rounds)
{
    arithmetic_triple_c = new DATATYPE[num_arithmetic_triples[rounds] ];
}

void init_beaverC_boolean(int rounds)
{
    boolean_triple_c = new DATATYPE[num_boolean_triples[rounds] ];
}


template <typename LayerParameter>
void deinit_LayerAB(DATATYPE** x, DATATYPE** w, std::vector<LayerParameter> p)
{
    for(int i = 0; i < p.size(); i++)
    {
#if PARTY == 0 || A_KNOWN == 0 // Party0 holds W in plain in AB2 setting
        delete[] w[i];
#endif
#if PARTY == 1 || A_KNOWN == 0 // Party 0 does not need X triples in AB2 setting
        delete[] x[i];
#endif
    }
    delete[] w;
    delete[] x;
}

void init_ConvAB()
{
    conv_triple_w = new DATATYPE*[conv_triple_params.size()]; 
    conv_triple_x = new DATATYPE*[conv_triple_params.size()];
}

void init_BatchNorm2DAB()
{
    bc2D_triple_w = new DATATYPE*[bc2D_triple_params.size()];
    bc2D_triple_x = new DATATYPE*[bc2D_triple_params.size()];
}

void init_FullyConnectedAB()
{
    fc_triple_w = new DATATYPE*[fc_triple_params.size()];
    fc_triple_x = new DATATYPE*[fc_triple_params.size()];
}

void init_ConvC()
{
    conv_triple_y = new DATATYPE[num_conv_c_triples];
}

#if CHEETAH_CONV_ASYNC_ACTIVE
// Starts the batched conv triples (generateLayerDummyTriples' conv_batched branch) on their own thread, before the ABY2
// preprocessing pass: the HE pipeline takes layer i once SetupConv2dTriples has recorded its masks, while the pass goes
// on. Same calls, same order and the same PRNG streams as after the pass, so the triples are unchanged.
void conv_async_start(std::string* ips, int base_port, int process_offset)
{
    if (conv_triple_params.empty())
        return;
    init_ConvC();
    std::vector<Utils::ConvParm> parms;
    for (const auto& p : conv_triple_params)
        parms.push_back(Utils::ConvParm{.batchsize = p.batchSize, .ic = p.din, .iw = p.inw, .ih = p.inh, .fc = p.din,
                                        .fw = p.ww, .fh = p.wh, .n_filters = p.dout, .stride = p.stride,
                                        .padding = p.padding});
    const std::string ip = ips[0];
    const int port = base_port + process_offset + CHEETAH_PORT_OFFSET;
    conv_async::launched = true;
    conv_async::worker = std::thread([parms, ip, port] {
        Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
        auto& keys = Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET);
        Iface::generateConvTriplesPackedBatch(keys, parms,
                                              A_KNOWN == 0 || PARTY == 1 ? (UINT_TYPE**) conv_triple_x : nullptr,
                                              A_KNOWN == 0 || PARTY == 0 ? (UINT_TYPE**) conv_triple_w : nullptr,
                                              (UINT_TYPE*) conv_triple_y, CHEETAH_PARTY, CHEETAH_THREADS,
                                              A_KNOWN == 0 ? Utils::PROTO::AB : Utils::PROTO::AB2, conv_async::wait_ready);
        conv_async::done = true;
    });
}
#endif

#if CHEETAH_CONV_EARLY_ACTIVE
// The OT phase (protocol_executer.hpp run_ot_phase), run by the mask-only forward once the convs have recorded their
// inputs and the conv triples have started
inline std::function<void()> g_early_ot_hook;
inline uint64_t g_early_conv_index = 0;  // convs recorded by the mask-only forward

// The conv triples of all layers, on four channels of their own (Keys::get_side_ios) next to the OT packs, which keep
// the regular channels; the inputs were recorded by the mask-only forward, so no layer waits. Same calls, same PRNG
// streams: the triples are the ones the pass would have started.
void conv_early_start(std::string* ips, int base_port, int process_offset)
{
    if (conv_triple_params.empty())
        return;
    init_ConvC();
    std::vector<Utils::ConvParm> parms;
    for (const auto& p : conv_triple_params)
        parms.push_back(Utils::ConvParm{.batchsize = p.batchSize, .ic = p.din, .iw = p.inw, .ih = p.inh, .fc = p.din,
                                        .fw = p.ww, .fh = p.wh, .n_filters = p.dout, .stride = p.stride,
                                        .padding = p.padding});
    const std::string ip = ips[0];
    const int port = base_port + process_offset + CHEETAH_PORT_OFFSET;
    conv_async::launched = true;
    // They are to use the cores the OT phase leaves idle: CONV_EARLY_THREADS threads while it runs (default half the
    // CHEETAH threads: with all of them the OT phase slowed down by about as much as the conv triples gained), all
    // CHEETAH threads after it (g_early_ot_hook lifts the limit; with half, the Zen 3 builds with short OT phases had
    // the conv triples as their tail); CONV_EARLY_NICE their nice value (inherited by the threads they start; default
    // 0: with 10-19 they became the tail of the PPA4 builds)
    const int nice_value = getenv("CONV_EARLY_NICE") ? atoi(getenv("CONV_EARLY_NICE")) : CHEETAH_CONV_EARLY_NICE;
    const int threads = getenv("CONV_EARLY_THREADS") ? std::max(4, atoi(getenv("CONV_EARLY_THREADS"))) : CHEETAH_CONV_EARLY_THREADS;
    Iface::conv_threads_now() = (size_t) threads;
    conv_async::worker = std::thread([parms, ip, port, nice_value] {
        if (nice_value != 0)
            setpriority(PRIO_PROCESS, (id_t) syscall(SYS_gettid), nice_value);
        auto& keys = Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET);
        Iface::generateConvTriplesPackedBatch(keys, parms,
                                              A_KNOWN == 0 || PARTY == 1 ? (UINT_TYPE**) conv_triple_x : nullptr,
                                              A_KNOWN == 0 || PARTY == 0 ? (UINT_TYPE**) conv_triple_w : nullptr,
                                              (UINT_TYPE*) conv_triple_y, CHEETAH_PARTY, CHEETAH_THREADS,
                                              A_KNOWN == 0 ? Utils::PROTO::AB : Utils::PROTO::AB2, nullptr,
                                              keys.get_side_ios(4));
        conv_async::done = true;
    });
}
#endif

void init_BatchNorm2DC()
{
    bc2D_triple_y = new DATATYPE[num_bc2D_c_triples];
}

void init_FullyConnectedC()
{
    fc_triple_y = new DATATYPE[num_fc_c_triples];
}

void deinit_ConvAB()
{
    deinit_LayerAB(conv_triple_x, conv_triple_w, conv_triple_params);
}

void deinit_ConvC()
{
    delete[] conv_triple_y;
}

void deinit_BatchNorm2DAB()
{
    deinit_LayerAB(bc2D_triple_x, bc2D_triple_w, bc2D_triple_params);
}

void deinit_BatchNorm2DC()
{
    delete[] bc2D_triple_y;
}

void deinit_FullyConnectedAB()
{
    deinit_LayerAB(fc_triple_x, fc_triple_w, fc_triple_params);
}

void deinit_FullyConnectedC()
{
    delete[] fc_triple_y;
}

void init_beaverAB2(int rounds)
{
#if PARTY == 0 // P0 holds a in plain in AB2 setting
    arithmetic_ab2_triple_a = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
    boolean_ab2_triple_a = new DATATYPE[num_ab2_boolean_triples[rounds] ];
#endif
#if PARTY == 1 // P0 doesn't need B1 for AB2
    arithmetic_ab2_triple_b = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
    boolean_ab2_triple_b = new DATATYPE[num_ab2_boolean_triples[rounds] ];
#endif
    // std::cout << "Initialized beaver AB2 for round " + std::to_string(rounds) + " with " + std::to_string(num_ab2_arithmetic_triples[rounds] * DATTYPE/BITLENGTH) + " arithmetic triples and " + std::to_string(num_ab2_boolean_triples[rounds] * DATTYPE) + " boolean triples.\n";
}

void init_beaverAB2_arithmetic(int rounds)
{
#if PARTY == 0 // P0 holds a in plain in AB2 setting
    arithmetic_ab2_triple_a = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
#endif
#if PARTY == 1 // P0 doesn't need B1 for AB2
    arithmetic_ab2_triple_b = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
#endif
}

void init_beaverAB2_boolean(int rounds)
{
#if PARTY == 0 // P0 holds a in plain in AB2 setting
    boolean_ab2_triple_a = new DATATYPE[num_ab2_boolean_triples[rounds] ];
#endif
#if PARTY == 1 // P0 doesn't need B1 for AB2
    boolean_ab2_triple_b = new DATATYPE[num_ab2_boolean_triples[rounds] ];
#endif
}

void init_beaverAB2C(int rounds)
{
    if(num_ab2_arithmetic_triples[rounds] > 0)
        arithmetic_ab2_triple_c = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
    if(num_ab2_boolean_triples[rounds] > 0)
        boolean_ab2_triple_c = new DATATYPE[num_ab2_boolean_triples[rounds] ];
    // std::cout << "Initialized beaver AB2 C for round " + std::to_string(rounds) + " with " + std::to_string(num_ab2_arithmetic_triples[rounds] * DATTYPE/BITLENGTH) + " arithmetic triples and " + std::to_string(num_ab2_boolean_triples[rounds] * DATTYPE) + " boolean triples.\n";
}

void init_beaverAB2C_arithmetic(int rounds)
{
    if(num_ab2_arithmetic_triples[rounds] > 0)
        arithmetic_ab2_triple_c = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
}

void init_beaverAB2C_boolean(int rounds)
{
    if(num_ab2_boolean_triples[rounds] > 0)
        boolean_ab2_triple_c = new DATATYPE[num_ab2_boolean_triples[rounds] ];
}
#else
void init_beaver()
{
    /* arithmetic_triple_index = 0; */
    /* boolean_triple_index = 0; */
    arithmetic_triple_a = new DATATYPE[total_arithmetic_triples_num];
    arithmetic_triple_b = new DATATYPE[total_arithmetic_triples_num];
    arithmetic_triple_c = new DATATYPE[total_arithmetic_triples_num];
    boolean_triple_a = new DATATYPE[total_boolean_triples_num];
    boolean_triple_b = new DATATYPE[total_boolean_triples_num];
    boolean_triple_c = new DATATYPE[total_boolean_triples_num];

    arithemtic_ab2_triple_a = new DATATYPE[total_ab2_arithmetic_triples_num];
    arithmetic_ab2_triple_b = new DATATYPE[total_ab2_arithmetic_triples_num];
    arithmetic_ab2_triple_c = new DATATYPE[total_ab2_arithmetic_triples_num];
    boolean_ab2_triple_a = new DATATYPE[total_ab2_boolean_triples_num];
    boolean_ab2_triple_b = new DATATYPE[total_ab2_boolean_triples_num];
    boolean_ab2_triple_c = new DATATYPE[total_ab2_boolean_triples_num];
}
#endif

void deinit_beaverAB2()
{
    // print("Deleting beaver AB2 arrays.");
#if PARTY == 0 
    delete[] arithmetic_ab2_triple_a;
    delete[] boolean_ab2_triple_a;
#elif PARTY == 1
    delete[] arithmetic_ab2_triple_b;
    delete[] boolean_ab2_triple_b;
#endif
}

void deinit_beaverAB2_arithmetic()
{
#if PARTY == 0 
    delete[] arithmetic_ab2_triple_a;
#elif PARTY == 1
    delete[] arithmetic_ab2_triple_b;
#endif
}

void deinit_beaverAB2_boolean()
{
#if PARTY == 0
    delete[] boolean_ab2_triple_a;
#elif PARTY == 1
    delete[] boolean_ab2_triple_b;
#endif
}

void deinit_beaverAB2C()
{
    // print("Deleting beaver AB2 C arrays.");
    if(arithmetic_ab2_triple_c != nullptr) 
    {
        delete[] arithmetic_ab2_triple_c;
        arithmetic_ab2_triple_c = nullptr;
    }
    if(boolean_ab2_triple_c != nullptr)
    {
        delete[] boolean_ab2_triple_c;
        boolean_ab2_triple_c = nullptr;
    }
}

void deinit_beaverAB2C_arithmetic()
{
    if(arithmetic_ab2_triple_c != nullptr) 
    {
        delete[] arithmetic_ab2_triple_c;
        arithmetic_ab2_triple_c = nullptr;
    }
}

void deinit_beaverAB2C_boolean()
{
    if(boolean_ab2_triple_c != nullptr)
    {
        delete[] boolean_ab2_triple_c;
        boolean_ab2_triple_c = nullptr;
    }
}


void deinit_beaverAB()
{
    // std::cout << "Deleting beaver AB arrays." << std::endl;
    delete[] arithmetic_triple_a;
    delete[] arithmetic_triple_b;
    delete[] boolean_triple_a;
    delete[] boolean_triple_b;
}

void deinit_beaverAB_arithmetic()
{
    delete[] arithmetic_triple_a;
    delete[] arithmetic_triple_b;
}

void deinit_beaverAB_boolean()
{
    delete[] boolean_triple_a;
    delete[] boolean_triple_b;
}

void deinit_beaverC()
{
    // std::cout << "Deleting beaver C arrays." << std::endl;
    if(arithmetic_triple_c != nullptr) 
    {
        delete[] arithmetic_triple_c;
        arithmetic_triple_c = nullptr;
    }
    if(boolean_triple_c != nullptr)
    {
        delete[] boolean_triple_c;
        boolean_triple_c = nullptr;
    }
}

void deinit_beaverC_arithmetic()
{
    if(arithmetic_triple_c != nullptr) 
    {
        delete[] arithmetic_triple_c;
        arithmetic_triple_c = nullptr;
    }
}

void deinit_beaverC_boolean()
{
    if(boolean_triple_c != nullptr)
    {
        delete[] boolean_triple_c;
        boolean_triple_c = nullptr;
    }
}

struct timespec k1, k2;

void generate_beaver_triples(std::string ips[], int base_port, int process_offset, uint64_t num_arith_triples, uint64_t num_bool_triples, std::string triple_type)
{
    uint64_t l_num_arithmetic_triples = num_arith_triples * DATTYPE / BITLENGTH;
    uint64_t l_num_boolean_triples = num_bool_triples * DATTYPE;
    uint64_t l_num_multiplexer_triples = num_multiplexer_triples * DATTYPE / BITLENGTH;
    uint64_t l_num_cot_triples = num_cot_triples * DATTYPE / BITLENGTH;
    uint64_t l_num_boolean_addition_triples = num_boolean_addition_triples * DATTYPE;
    uint64_t l_num_beaver_3_tuples = num_beaver_3_tuples * DATTYPE;
    uint64_t l_num_beaver_4_tuples = num_beaver_4_tuples * DATTYPE;
    uint64_t l_num_random_multiplications = num_random_multiplications * DATTYPE;

#if FAKE_TRIPLES == 1
    print("Fake Triples set to 1, generating fake triples ... \n");
#else
    // print("Generating ", triple_type.data(), "  Triples ... \n");
    print("Generating %s Triples ... \n", triple_type.c_str());
#endif
    clock_t time_beaver_function_start = clock();
    clock_gettime(CLOCK_REALTIME, &k1);
    std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();

#if num_players == 2
if(triple_type == "LXLY") {
    generateArithmeticTriples(arithmetic_triple_a,
                              arithmetic_triple_b,
                              arithmetic_triple_c,
                              BITLENGTH,
                              l_num_arithmetic_triples,
                              ips[0],
                              base_port + process_offset);
    generateBooleanTriples(boolean_triple_a,
                           boolean_triple_b,
                           boolean_triple_c,
                           BITLENGTH,
                           l_num_boolean_triples,
                           ips[0],
                           base_port + process_offset);
} else if(triple_type == "LXLY2") {
    generateArithmeticAB2Triples(arithmetic_ab2_triple_a,
                                 arithmetic_ab2_triple_b,
                                 arithmetic_ab2_triple_c,
                                 BITLENGTH,
                                 l_num_arithmetic_triples,
                                 ips[0],
                                 base_port + process_offset);
    generateBooleanAB2Triples(boolean_ab2_triple_a,
                                boolean_ab2_triple_b,
                                boolean_ab2_triple_c,
                                BITLENGTH,
                                l_num_boolean_triples,
                                ips[0],
                                base_port + process_offset);

} 
else if (triple_type == "CONV") {
    generateConvTriples(conv_triple_w,
                        conv_triple_x,
                        conv_triple_y,
                        BITLENGTH,
                        conv_triple_params,
                        ips[0],
                        base_port + process_offset);
}
else if (triple_type == "FC") {
    generateFCTriples(fc_triple_w,
                     fc_triple_x,
                     fc_triple_y,
                     BITLENGTH,
                     fc_triple_params,
                     ips[0],
                     base_port + process_offset);
}
else if (triple_type == "BATCHNORM2D") {
    generateBatchNorm2DTriples(bc2D_triple_w,
                              bc2D_triple_x,
                              bc2D_triple_y,
                              BITLENGTH,
                              bc2D_triple_params,
                              ips[0],
                              base_port + process_offset);
}
#if A2B_ONLINE_OPT == 1
else if (triple_type == "BOOLEANADDITION") {
    generateBooleanAdditionTriples(boolean_addition_triple_a,
                                   boolean_addition_triple_b,
                                   boolean_addition_triple_c,
                                   BITLENGTH,
                                   l_num_boolean_addition_triples,
                                   ips[0],
                                   base_port + process_offset);
}
#endif
#if BIT_INJECTION_PREPROCESSING_OPT == 1
else if (triple_type == "MULTIPLEXER") {
    generateMultiplexerTriples(multiplexer_triple_a,
                               multiplexer_triple_b,
                               multiplexer_triple_c,
                               BITLENGTH,
                               l_num_multiplexer_triples,
                               ips[0],
                               base_port + process_offset);
}
else if (triple_type == "COT") {
    generateCOTTriples(cot_triple_a,
                       cot_triple_c,
                       BITLENGTH,
                       l_num_cot_triples,
                       ips[0],
                       base_port + process_offset);
}
#endif
#if BEAVER_N_TUPLES == 1
else if (triple_type == "BEAVER_N_TUPLES") {
    generateBeaverNDummyTuples(beaver_3_tuples, beaver_4_tuples, l_num_beaver_3_tuples, l_num_beaver_4_tuples, ips[0], base_port + process_offset);
}
#endif
else if (triple_type == "RANDOM_MULTIPLICATION") {
    generateRandomMultiplications(random_multiplication_a, random_multiplication_b, l_num_random_multiplications, ips[0], base_port + process_offset);
}
else {
    std::cerr << "Unknown triple type: " << triple_type << std::endl;
    exit(1);
}
#else
    std::cerr << "Beaver triples not implemented for more than 2 parties" << std::endl;
    exit(1);
#endif

    clock_gettime(CLOCK_REALTIME, &k2);
    double accum_beaver = (k2.tv_sec - k1.tv_sec) + (double)(k2.tv_nsec - k1.tv_nsec) / (double)1000000000L;
    clock_t time_beaver_function_finished = clock();
    print("Time measured to perform beaver triple generation clock: %fs \n",
          double((time_beaver_function_finished - time_beaver_function_start)) / CLOCKS_PER_SEC);
    print("Time measured to perform beaver triple generation getTime: %fs \n", accum_beaver);
    print("Time measured to perform beaver triple generation chrono: %fs \n",
          double(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - p)
                     .count()) /
              1000000);


}

void print_num_triples()
{
#if PRINT_IMPORTANT == 1
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Arithmetic Beaver Triples Required: " << total_arithmetic_triples_num * DATTYPE / BITLENGTH
              << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Boolean Beaver Triples Required: " << total_boolean_triples_num * DATTYPE << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Arithmetic AB2 Beaver Triples Required: " << total_ab2_arithmetic_triples_num * DATTYPE / BITLENGTH
              << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
                << "Boolean AB2 Beaver Triples Required: " << total_ab2_boolean_triples_num * DATTYPE << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Boolean Addition Triples Required: " << num_boolean_addition_triples * DATTYPE << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Multiplexer Triples Required: " << num_multiplexer_triples * DATTYPE / BITLENGTH << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": " 
              << "COT Triples Required: " << num_cot_triples * DATTYPE / BITLENGTH << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Beaver 3-Tuples Required: " << num_beaver_3_tuples * DATTYPE << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Beaver 4-Tuples Required: " << num_beaver_4_tuples * DATTYPE << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Random Multiplications Required: " << num_random_multiplications * DATTYPE << std::endl;
#if A_KNOWN == 0
    std::string triple_type_str = "AB";
#else
    std::string triple_type_str = "AB2";
#endif
    for(int i = 0; i < conv_triple_params.size(); i++)
    {
        std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
            << "Convolution " << triple_type_str << " Triples Required for Conv layer " << i << ": " 
                  << conv_triple_params[i].batchSize * (((conv_triple_params[i].out_h + 0) / 1) * (((conv_triple_params[i].out_w + 0) / 1)) * conv_triple_params[i].dout)
                  << std::endl;
    }
    for(int i = 0; i < fc_triple_params.size(); i++)
    {
        std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
                  << "Fully Connected " << triple_type_str << " Triples Required for FC layer " << i << ": " 
                  << fc_triple_params[i].out_feat * fc_triple_params[i].batchSize
                  << std::endl;
    }
    for(int i = 0; i < bc2D_triple_params.size(); i++)
    {
        std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
                  << "BatchNorm2D " << triple_type_str << " Triples Required for BN2D layer " << i << ": " 
                  << bc2D_triple_params[i].batchSize * bc2D_triple_params[i].ch * bc2D_triple_params[i].h * bc2D_triple_params[i].w
                  << std::endl;
    }
#endif
}
