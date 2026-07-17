#pragma once
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
    return false;
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
#define RESHARE_BAKE_ACTIVE \
    (RESHARE_OPT == 1 && RESHARE_OPT_SIM == 1 && DATTYPE == BITLENGTH && \
     (RCA_MSB == 1 || PPA_MSB == 1 || PPA4_MSB == 1))

// CUT_FRACTIONAL_BITS_OPT (docs/CUT_FRACTIONAL_BITS_OPT.md): compile-time eligibility. Under
// TRUNC_DELAYED == 0 the ReLU input is freshly truncated, so its value fits BITLENGTH-FRACTIONAL
// signed bits and the MSB adder's top FRACTIONAL slices are redundant. Whether a given adder
// instance actually applies the cut is the RUNTIME flag g_cut_frac_active (set by RELU only -
// max/min/comparison adders run the full circuit on the same build).
// Currently implemented for the RESHARE_OPT=1 generated circuits of RCA, PPA, and PPA4 (k = 32);
// other circuit variants safely no-op the cut until they are patched too.
// Second leg: the A2B bake (A2B_ONLINE_OPT + A_KNOWN_TO_EVALUATORS_OPT) also supports the cut - the
// a_ab adders skip their top-FRACTIONAL-slice gates internally (RCA + PPA; the PPA4 a_ab adder has no
// cut guards yet and simply computes fully, which stays correct because under the bake no external
// stream count depends on adder-internal gates: [c], S1 and the boolean addition stay full-width).
#define CUT_FRAC_ELIGIBLE \
    (CUT_FRACTIONAL_BITS_OPT == 1 && TRUNC_DELAYED == 0 && FRACTIONAL >= 1 && FRACTIONAL <= BITLENGTH - 3 && \
     (RESHARE_OPT == 1 || (A2B_ONLINE_OPT == 1 && A_KNOWN_TO_EVALUATORS_OPT == 1)) && \
     ROT_PREPROCESSING_OPT == 1 && BITLENGTH == 32 && \
     (RCA_MSB == 1 || PPA_MSB == 1 || PPA4_MSB == 1))
#define CUT_FRAC_ELIGIBLE_PPA4 (CUT_FRAC_ELIGIBLE && PPA4_MSB == 1)

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
#define A2B_CONV_BAKE_ACTIVE (A2B_ONLINE_OPT == 1 && A2B_CONV_BAKE == 1 && DATTYPE == BITLENGTH)
#if A2B_CONV_BAKE_ACTIVE
#include <vector>
std::vector<DATATYPE> g_a2b_ia;   // this party's random boolean A2B-mask slices (boolean-adder input)
std::vector<DATATYPE> g_a2b_lz;   // derived conv mask -untranspose(ia); ortho(-lz) == ia
std::vector<DATATYPE> g_a2b_c;    // this party's share of [c] = bool(-lz), from the early boolean addition
uint64_t g_a2b_layer_base = 0;    // g_a2b_lz base for the current layer's A2B group (reset per phase)
uint64_t g_a2b_c_cursor = 0;      // A2B-S2 [c] cursor (reset per phase)
// init_a2b_bake / a2b_bake_store_c are defined further down, after the boolean_addition_triple buffers.

// Index-addressed (NOT a linear cursor): the conv mask for the e-th layer-local output goes to
// g_a2b_lz[layer_base + g_bake_batch_offset + e]. e is the C[] output position within the batch
// element (mask/send index / bake_index), g_bake_batch_offset the batch-element base (set by the
// conv/FC forward), g_a2b_layer_base the layer base (snapped to the [c] group boundary after each
// A2B). This matches the A2B, which packs C[] linearly into sints and consumes [c] slice-per-position,
// even when the conv mask/send is called in tiled (non-linear) order.
template <typename Datatype, typename func_sub>
inline Datatype a2b_bake_conv_mask(uint64_t e, func_sub SUB)
{
    const uint64_t idx = g_a2b_layer_base + g_bake_batch_offset + e;
    // Out of range == this output never feeds an A2B (g_a2b_lz covers exactly the INIT-counted A2B
    // slices; e.g. the network's final FC before the reveal). Fall back to a fresh synced PRNG draw -
    // the baseline behavior. Returning a constant here instead would make P1's r1 = -low TINY and
    // break the SecureML trunc wrap (B >= |v| fails) -> every negative output off by +2^(K-F).
    if (idx >= g_a2b_lz.size())
        return getRandomVal(PSELF);
    Datatype lz = g_a2b_lz[idx];
    // Bias pre-compensation: add_bias later shifts this output's mask by the party's OWN bias-mask
    // share (owner: get_mask() of its b share; non-owner: 0 -> no-op, so P1's trunc-image constraint
    // is untouched). Subtract it here so the TOTAL mask after add_bias equals the committed lz that
    // [c] was built for. Same buffer/indexing as the reshare bake (g_bake_bias_l, batch-local e).
    if (g_bake_bias_l != nullptr && g_bake_bias_len > 0)
        lz = SUB(lz, g_bake_bias_l[(g_bake_batch_offset + e) % g_bake_bias_len]);
    return lz;
}

// [c] share for the next A2B-S2 slice - identical in PRE and LIVE, so the msb adder's beaver triples
// (generated in PRE from this out.l) match what LIVE consumes.
inline DATATYPE a2b_bake_get_c()
{
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
#if CUT_FRAC_ELIGIBLE
    return k == BITLENGTH && i < FRACTIONAL;
#else
    (void) k; (void) i;
    return false;
#endif
}

constexpr bool cut_frac_identity(int k, int i)  // leaf (g,p) := (0,1) substituted slice?
{
#if CUT_FRAC_ELIGIBLE
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
    if (k == 32)
        return (uint64_t)(24 - cut_frac_ppa4_b3_skipped_below(24));
    return k == 16 ? 9 : 4;
}

// P0-side helper for the PPA4 SIM=1 zero_add skip: counts prepare_A2B_S1 calls since the last
// beaver-3-tuple retrieval, so slice masks can be peeked at the tuple positions the group's adder
// WILL consume (all groups are prepared before any adder is constructed). Reset in
// retrieveBeaver3Tuple (any retrieval means the S1 batch has ended).
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
template <typename Datatype, typename func_sub>
inline void bake_reshare_mask(Datatype& l, int bake_index, func_sub SUB)
{
#if PARTY == 1 && RESHARE_BAKE_ACTIVE  // P1-ONLY: baking P0's mask online would desync its PRE vs LIVE masks
    constexpr int K = BITLENGTH;
    constexpr uint64_t R = reshares_per_adder(K);
    const uint64_t e = (uint64_t) bake_index + g_bake_batch_offset;  // batch-global output index
    const uint64_t g = e / K;        // bit-sliced A2B group (one adder per group)
    const int j = (int) (e % K);     // value's word index in the group -> lane-bit (K-1-j) after the transpose
    const uint64_t base = curr_random_multiplication_index + g * R;
    if (base + R > num_random_multiplications)
        return;  // this layer's outputs never reach an MSB adder (e.g. final layer) - leave the mask random
    UINT_TYPE negl = (UINT_TYPE) l;
    for (int i = 1; i < K; i++)  // slice 0 (numeric MSB) is never reshared
    {
        if (cut_frac_identity(K, i))
            continue;  // CUT_FRACTIONAL_BITS_OPT: identity-substituted slice, not reshared in the circuit
        const int t = reshare_rt_offset(K, i);
        if (t < 0)
            continue;
        const UINT_TYPE rta = (UINT_TYPE) random_multiplication_a[base + (uint64_t) t];
        const UINT_TYPE bit = (rta >> (K - 1 - j)) & (UINT_TYPE) 1;
        const int nb = K - 1 - i;  // numeric bit position of slice i
        negl = (negl & ~((UINT_TYPE) 1 << nb)) | (bit << nb);
    }
#if PPA4_MSB == 1
    // PPA4 additionally SIM-skips the input-wire zero_adds: bake our (P1-local, see party_local_bc
    // in the tuple generation) beaver3 .c fields into the zero_added b-wire slices so the skipped
    // re-masking would have been a no-op. Same base-offset reasoning as the random multiplications.
    const uint64_t b3_base = curr_beaver_3_triple_index + g * b3_tuples_per_adder(K);
    if (b3_base + b3_tuples_per_adder(K) <= num_beaver_3_tuples)
    {
        for (int i = 1; i < K; i++)
        {
            if (cut_frac_identity(K, i))
                continue;  // CUT_FRACTIONAL_BITS_OPT: identity-substituted slice, zero_add skipped
            const int t3 = ppa4_zero_add_t3(K, i);
            if (t3 < 0)
                continue;
            const UINT_TYPE c3 = (UINT_TYPE) beaver_3_tuples.c[b3_base + (uint64_t) t3];
            const UINT_TYPE bit = (c3 >> (K - 1 - j)) & (UINT_TYPE) 1;
            const int nb = K - 1 - i;
            negl = (negl & ~((UINT_TYPE) 1 << nb)) | (bit << nb);
        }
    }
#endif
    Datatype l_new = SUB(SET_ALL_ZERO(), (Datatype) negl);
    if (g_bake_bias_l != nullptr && g_bake_bias_len > 0)  // pre-compensate a shared bias added after the GEMM
        l_new = SUB(l_new, g_bake_bias_l[e % g_bake_bias_len]);
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
    const Datatype m1 = a2b_bake_conv_mask<Datatype>((uint64_t)(bake_index < 0 ? 0 : bake_index), SUB);
    const UINT_TYPE low = (UINT_TYPE) getRandomVal(PSELF) & (((UINT_TYPE) 1 << FRACTIONAL) - (UINT_TYPE) 1);
    return (Datatype) (UINT_TYPE) (0 - (((UINT_TYPE) m1 << FRACTIONAL) + low));
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
// P1-side live check for the SIM=1 reshare condition (see the counter comment above). Registers a
// one-line summary at exit; the first few mismatches are printed with their rt index for diagnosis.
template <typename Datatype>
inline void reshare_sim_check(Datatype l, Datatype mask)
{
#if PARTY == 1 && DATTYPE == BITLENGTH
    if (current_phase != PHASE_LIVE)
        return;
    static bool reg = []()
    {
        atexit([]() {
            fprintf(stderr, "RB-FINAL checks=%llu mismatch=%llu\n", (unsigned long long) g_rb_checks,
                    (unsigned long long) g_rb_mismatch);
        });
        return true;
    }();
    (void) reg;
    g_rb_checks++;
    if ((UINT_TYPE) l != (UINT_TYPE) mask)
    {
        g_rb_mismatch++;
        if (g_rb_mismatch <= 4)
            fprintf(stderr, "RB-MISMATCH #%llu idx=%llu l=%08x mask=%08x\n", (unsigned long long) g_rb_mismatch,
                    (unsigned long long) (curr_random_multiplication_index - 1), (unsigned) (UINT_TYPE) l,
                    (unsigned) (UINT_TYPE) mask);
    }
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
#if RANDOM_ALGORITHM == 2 && USE_SSL_AES == 0
    AES_TYPE saved_counter = aes_counter[PSELF];
    uint64_t saved_numgen = num_generated[PSELF];
#endif
    for (uint64_t base = 0; base + K <= num_slices; base += K)
    {
        Datatype ia[K];
        for (int i = 0; i < K; i++) { ia[i] = getRandomVal(PSELF); g_a2b_ia[base + i] = ia[i]; }
        // real_ortho (unorthogonalize_boolean) is self-inverse, so with lz = -untranspose(ia): ortho(-lz)==ia.
        alignas(sizeof(Datatype)) UINT_TYPE t2[DATTYPE];
        Datatype tmp[K];
        for (int i = 0; i < K; i++) tmp[i] = ia[i];
        unorthogonalize_boolean(tmp, t2);
#if TRUNC_DELAYED == 0 && PARTY == 1
        // TD=0: P1's conv/FC output mask is l1 = TRUNC(-r1), and TRUNC (FUNC_TRUNC = OP_TRUNC under
        // SKIP_PRE=0) is a LOGICAL shift - its image has the top FRACTIONAL bits ZERO. Constrain the
        // committed m1 the same way (zero the top F bits; NOT sign-extension) and RE-derive
        // ia = bool(-m1) so the early boolean addition still yields [c] = bool(-(lz0+m1)). The remask
        // path (no trunc) also uses this m1 - a validly-masked, just constrained, value - stays correct.
        for (int i = 0; i < K; i++)
        {
            UINT_TYPE mi = (UINT_TYPE) SUB(SET_ALL_ZERO(), (Datatype) t2[i]);      // m1 = -t2 (numeric)
            mi &= (((UINT_TYPE) 1 << (BITLENGTH - FRACTIONAL)) - (UINT_TYPE) 1);   // logical-trunc image
            g_a2b_lz[base + i] = (Datatype) mi;
            t2[i] = (UINT_TYPE) (0 - mi);                                          // t2 := -m1 (for ia)
        }
        Datatype ia_new[K];
        orthogonalize_boolean(t2, ia_new);  // ia = bool(-lz) = bool(-m1)
        for (int i = 0; i < K; i++) g_a2b_ia[base + i] = ia_new[i];
#else
        for (int i = 0; i < K; i++) g_a2b_lz[base + i] = SUB(SET_ALL_ZERO(), (Datatype) t2[i]);
#endif
    }
#if RANDOM_ALGORITHM == 2 && USE_SSL_AES == 0
    aes_counter[PSELF] = saved_counter;
    num_generated[PSELF] = saved_numgen;
#endif
    // Hand this party's ia to the boolean-addition input buffer (P0 -> a, P1 -> b), in slice order.
    for (uint64_t e = 0; e < num_slices; e++)
#if PARTY == 0
        boolean_addition_triple_a[e] = g_a2b_ia[e];
#else
        boolean_addition_triple_b[e] = g_a2b_ia[e];
#endif
}

// After the early boolean addition has produced boolean_addition_triple_c, capture this party's [c] share.
inline void a2b_bake_store_c(uint64_t num_slices)
{
    for (uint64_t e = 0; e < num_slices && e < g_a2b_c.size(); e++)
        g_a2b_c[e] = boolean_addition_triple_c[e];
}
#endif

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
    curr_arithmetic_triple_index++;
    return triple<Datatype>{arithmetic_triple_a[curr_arithmetic_triple_index - 1],
                            arithmetic_triple_b[curr_arithmetic_triple_index - 1],
                            arithmetic_triple_c[curr_arithmetic_triple_index - 1]};
#endif
}

template <typename Datatype>
triple<Datatype> retrieveBooleanTriple()
{
#if SKIP_PRE == 1
    return triple<Datatype>{SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()};
#else
    curr_boolean_triple_index++;
    return triple<Datatype>{boolean_triple_a[curr_boolean_triple_index - 1],
                            boolean_triple_b[curr_boolean_triple_index - 1],
                            boolean_triple_c[curr_boolean_triple_index - 1]};
    /* return triple<Datatype>{boolean_triple_a[boolean_triple_index], boolean_triple_b[boolean_triple_index],
     * boolean_triple_c[boolean_triple_index++]}; */
#endif
}

template <typename Datatype>
Beaver3Tuple<Datatype> retrieveBeaver3Tuple()
{
    g_a2b_s1_pending = 0;  // an adder is consuming -> the prepare_A2B_S1 batch (if any) has ended
#if SKIP_PRE == 1
    return Beaver3Tuple<Datatype>{
        SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO(),
        SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()
    };
#else
    Beaver3Tuple<Datatype> tuple{
        beaver_3_tuples.a[curr_beaver_3_triple_index],
        beaver_3_tuples.b[curr_beaver_3_triple_index],
        beaver_3_tuples.c[curr_beaver_3_triple_index],
        beaver_3_tuples.ab[curr_beaver_3_triple_index],
        beaver_3_tuples.ac[curr_beaver_3_triple_index],
        beaver_3_tuples.bc[curr_beaver_3_triple_index],
        beaver_3_tuples.abc[curr_beaver_3_triple_index]
    };
    curr_beaver_3_triple_index++;
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
    Beaver4Tuple<Datatype> tuple{
        beaver_4_tuples.a[curr_beaver_4_triple_index],
        beaver_4_tuples.b[curr_beaver_4_triple_index],
        beaver_4_tuples.c[curr_beaver_4_triple_index],
        beaver_4_tuples.d[curr_beaver_4_triple_index],
        beaver_4_tuples.ab[curr_beaver_4_triple_index],
        beaver_4_tuples.ac[curr_beaver_4_triple_index],
        beaver_4_tuples.ad[curr_beaver_4_triple_index],
        beaver_4_tuples.bc[curr_beaver_4_triple_index],
        beaver_4_tuples.bd[curr_beaver_4_triple_index],
        beaver_4_tuples.cd[curr_beaver_4_triple_index],
        beaver_4_tuples.abc[curr_beaver_4_triple_index],
        beaver_4_tuples.abd[curr_beaver_4_triple_index],
        beaver_4_tuples.acd[curr_beaver_4_triple_index],
        beaver_4_tuples.bcd[curr_beaver_4_triple_index],
        beaver_4_tuples.abcd[curr_beaver_4_triple_index]
    };
    curr_beaver_4_triple_index++;
    return tuple;
#endif
}

template <typename Datatype>
RandomMultiplication<Datatype> retrieveRandomMultiplication()
{
#if SKIP_PRE == 1
    return RandomMultiplication<Datatype>{SET_ALL_ZERO(), SET_ALL_ZERO()};
#else
    RandomMultiplication<Datatype> tuple{
        random_multiplication_a[curr_random_multiplication_index],
        random_multiplication_b[curr_random_multiplication_index]
    };
    curr_random_multiplication_index++;
    return tuple;
#endif
}

template <typename Datatype>
triple<Datatype> retrieveArithmeticAB2Triple()
{
#if SKIP_PRE == 1
    return triple<Datatype>{SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()};
#else
    curr_arithmetic_ab2_triple_index++;
    return triple<Datatype>{arithmetic_ab2_triple_a[curr_arithmetic_ab2_triple_index - 1],
                            arithmetic_ab2_triple_b[curr_arithmetic_ab2_triple_index - 1],
                            arithmetic_ab2_triple_c[curr_arithmetic_ab2_triple_index - 1]};
#endif
}

template <typename Datatype>
triple<Datatype> retrieveBooleanAB2Triple()
{
#if SKIP_PRE == 1
    return triple<Datatype>{SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()};
#else
    curr_boolean_ab2_triple_index++;
    return triple<Datatype>{boolean_ab2_triple_a[curr_boolean_ab2_triple_index - 1],
                            boolean_ab2_triple_b[curr_boolean_ab2_triple_index - 1],
                            boolean_ab2_triple_c[curr_boolean_ab2_triple_index - 1]};
#endif
}
    
    template <typename Datatype>
void storeArithmeticABTriple(const Datatype a, const Datatype b)
{
    arithmetic_triple_a[arithmetic_triple_index] = a;
    arithmetic_triple_b[arithmetic_triple_index] = b;
    arithmetic_triple_index++;
}

template <typename Datatype>
void storeBooleanABTriple(const Datatype a, const Datatype b)
{
    boolean_triple_a[boolean_triple_index] = a;
    boolean_triple_b[boolean_triple_index] = b; //B1 is not needed for the AB2 protocol
    boolean_triple_index++;
}

    template <typename Datatype>
void storeArithmeticAB2Triple(const Datatype a, const Datatype b)
{
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
