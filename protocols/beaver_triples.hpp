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

// Reshare wiring of the *_and_ab_reshared adders: which bit-slice (adder wire index i, 0 = numeric MSB,
// k-1 = numeric LSB) is reshared with which random_triples[] offset within one adder. -1 = not reshared.
// Must mirror the generated circuit constructors (rca_msb / ppa_msb_unsafe / ppa_msb_4way _and_ab_reshared.hpp).
constexpr int reshare_rt_offset(int k, int i)
{
#if RCA_MSB == 1
    return (i == k - 1) ? 0 : -1;  // RCA reshares only the LSB slice (first carry gate), rt[0]
#elif PPA_MSB == 1
    return (i >= 1 && i < k) ? i - 1 : -1;  // PPA reshares slices 1..k-1 with rt[i-1], ascending
#elif PPA4_MSB == 1
    // PPA4 reshares the AND2 "generate" wires; retrieval order is circuit-specific (k=32: wire 22 is LAST).
    if (k == 32)
    {
        switch (i)
        {
            case 1: return 0; case 4: return 1; case 7: return 2; case 10: return 3; case 13: return 4;
            case 16: return 5; case 19: return 6; case 23: return 7; case 26: return 8; case 29: return 9;
            case 22: return 10;
            default: return -1;
        }
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
    return (uint64_t)(k - 1);
#elif PPA4_MSB == 1
    return k == 32 ? 11 : (k == 16 ? 5 : 3);
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
        switch (i)
        {
            case 2: return 0; case 5: return 2; case 8: return 4; case 11: return 6;
            case 14: return 8; case 17: return 10; case 20: return 12; case 24: return 14;
            case 27: return 16; case 30: return 18;
            default: return -1;
        }
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
constexpr uint64_t b3_tuples_per_adder(int k) { return k == 32 ? 24 : (k == 16 ? 9 : 4); }

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
    Datatype r1 = getRandomVal(PSELF);
#if RESHARE_BAKE_ACTIVE  // gated: consumes an extra PRNG draw
    if (bake_index >= 0)
        r1 = construct_mwk_r1_baked(r1, getRandomVal(PSELF), bake_index, SUB);
#endif
    return r1;
}

// Untruncated mask l = -r1 (TRUNC_DELAYED): fully bakeable, no image constraint.
template <typename Datatype, typename func_sub>
inline Datatype mwk_choose_r1_no_trunc(int bake_index, func_sub SUB)
{
    Datatype r1 = getRandomVal(PSELF);
    if (bake_index >= 0)
    {
        Datatype l_t = r1;
        bake_reshare_mask(l_t, bake_index, SUB);  // no-op unless RESHARE_BAKE_ACTIVE && PARTY == 1
        r1 = SUB(SET_ALL_ZERO(), l_t);
    }
    return r1;
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
