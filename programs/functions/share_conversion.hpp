#pragma once
#include <functional>
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"  // FloatFixedConverter (reciprocal for fused avg / delayed trunc)
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include "../../protocols/Protocols.h"
#include "stream_parallel.hpp"
#if ADDITIONAL_PPA_THREADS > 0
#include <thread>
#endif

#if RCA_MSB == 0 && PPA_MSB == 0 && PPA4_MSB == 0
#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
#ifndef RCA_MSB
#define RCA_MSB 1
#endif
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1
#ifndef PPA4_MSB
#define PPA4_MSB 1
#endif
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0
#ifndef PPA_MSB
#define PPA_MSB 1
#endif
#endif
#endif

#if ROT_PREPROCESSING_OPT == 1
#if A_KNOWN_TO_EVALUATORS_OPT == 1
#include "adders/zero_add_adders/rca_and_a_ab.hpp"
#define FULL_ADDER_TYPE RCA_A_AB
#elif RESHARE_OPT == 1
#include "adders/zero_add_adders/rca_and_ab_reshared.hpp"
#define FULL_ADDER_TYPE RCA_AB
#else
#include "adders/zero_add_adders/rca_and_ab.hpp"
#define FULL_ADDER_TYPE RCA_AB
#endif
#else
#define FULL_ADDER_TYPE BooleanAdder
#include "adders/rca.hpp"
#endif

#if ROT_PREPROCESSING_OPT == 1

#if PPA4_MSB == 1
// BITLENGTH 64: no a-known four-way circuit (the generated one needs the hand fixes of the 8/16/32-bit ones, see
// scripts/circuits/gen_64bit_adders.py); the AB circuit takes the public m as a share with mask 0
#if A_KNOWN_TO_EVALUATORS_OPT == 1 && BITLENGTH != 64
#if ADDITIONAL_PPA_THREADS > 0
#include "adders/zero_add_adders/ppa_msb_4way_and_a_ab_split.hpp"
#else
#include "adders/zero_add_adders/ppa_msb_4way_and_a_ab.hpp"
#endif
#define ADDER_TYPE PPA_MSB_4Way_A_AB
#elif RESHARE_OPT == 1 
#if ADDITIONAL_PPA_THREADS > 0
#include "adders/zero_add_adders/ppa_msb_4way_and_ab_reshared_split.hpp"
#else
#include "adders/zero_add_adders/ppa_msb_4way_and_ab_reshared.hpp"
#endif
#define ADDER_TYPE PPA_MSB_4Way_AB
#else
#include "adders/zero_add_adders/ppa_msb_4way_and_ab.hpp"
#define ADDER_TYPE PPA_MSB_4Way_AB
#endif
#endif

#if PPA_MSB == 1
#if A_KNOWN_TO_EVALUATORS_OPT == 1
#include "adders/zero_add_adders/ppa_msb_unsafe_and_a_ab.hpp"
#define ADDER_TYPE PPA_MSB_Unsafe_A_AB
#elif RESHARE_OPT == 1
#include "adders/zero_add_adders/ppa_msb_unsafe_and_ab_reshared.hpp"
#define ADDER_TYPE PPA_MSB_Unsafe_AB
#else
#include "adders/zero_add_adders/ppa_msb_unsafe_and_ab.hpp"
#define ADDER_TYPE PPA_MSB_Unsafe_AB
#endif
#endif

#if RCA_MSB == 1
#if A_KNOWN_TO_EVALUATORS_OPT == 1
#include "adders/zero_add_adders/rca_msb_and_a_ab.hpp"
#define ADDER_TYPE RCA_MSB_A_AB
#elif RESHARE_OPT == 1
#include "adders/zero_add_adders/rca_msb_and_ab_reshared.hpp"
#define ADDER_TYPE RCA_MSB_AB
#else
#include "adders/zero_add_adders/rca_msb_and_ab.hpp"
#define ADDER_TYPE RCA_MSB_AB
#endif
#endif

#else
#if (BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0) || RCA_MSB == 1
#define ADDER_TYPE BooleanAdder_MSB
#include "adders/rca_msb.hpp"
#elif (BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1) || PPA4_MSB == 1
#define ADDER_TYPE PPA_MSB_4Way
#include "adders/ppa_msb_4_way.hpp"
#elif (BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0) || PPA_MSB == 1
#define ADDER_TYPE PPA_MSB_Unsafe
#include "adders/ppa_msb_unsafe.hpp"
#endif
#endif
// The cut's narrow adders (CUT_FRAC_NARROW): the a-known four-way circuit has no narrow form (the generator's needs the
// hand fixes of the 8/16/32-bit ones), so A2bits PPA4 takes the AB circuit, the public m as a share of mask 0
#if PPA4_MSB == 1 && A_KNOWN_TO_EVALUATORS_OPT == 1 && ROT_PREPROCESSING_OPT == 1
#include "adders/zero_add_adders/ppa_msb_4way_and_ab.hpp"
#define NARROW_ADDER_TYPE PPA_MSB_4Way_AB
#else
#define NARROW_ADDER_TYPE ADDER_TYPE
#endif
#if TE_FUSED_ACTIVE
// TE's low adders (Ts1Range::te): the carry into bit FRACTIONAL, the MSB of an (F + 1)-bit a-known adder (RCA for RCA
// builds, else the prefix adder: no more rounds than the ReLU's); widths 8 and 16 are in the families' files, the
// others in low/ (scripts/circuits/gen_64bit_adders.py)
#if A_KNOWN_TO_EVALUATORS_OPT == 0 || ADDITIONAL_PPA_THREADS > 0
#error "TRUNC_APPROACH 2 / 3 with PROTOCOL 4: TE takes the a-known adders (A2bits: A_KNOWN_TO_EVALUATORS_OPT=1), not the split four-way ones"
#endif
#if RCA_MSB == 1
#include "adders/zero_add_adders/rca_msb_and_a_ab.hpp"
#include "adders/zero_add_adders/low/rca_msb_and_a_ab.hpp"
#define TE_LOW_ADDER_TYPE RCA_MSB_A_AB
#else
#include "adders/zero_add_adders/ppa_msb_unsafe_and_a_ab.hpp"
#include "adders/zero_add_adders/low/ppa_msb_unsafe_and_a_ab.hpp"
#define TE_LOW_ADDER_TYPE PPA_MSB_Unsafe_A_AB
#endif
#endif
// compute msbs of a range of arithemtic shares
template <typename D>
inline bool equal_word(const D& a, const D& b)
{
    return std::memcmp(&a, &b, sizeof(D)) == 0;
}

#if TS1_FUSED_ACTIVE || A2B_DCUT_ACTIVE
// The A2B input transform of the running ReLU (g_a2b_xform), online only: the other passes have no m
template <typename Datatype, typename Share>
void a2b_xform_input(sint_t<Additive_Share<Datatype, Share>>* val, int len)
{
    if constexpr (requires(Share& s) { s.raw_m(); })
        stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) {
            auto* sh = val[i].get_share_pointer();
            for (int j = 0; j < BITLENGTH; j++)
            {
                Datatype& m = sh[j].raw_m();
                const uint64_t v = (uint64_t) i * BITLENGTH + j;
                switch (g_a2b_xform)
                {
#if TS1_FUSED_ACTIVE
#if TS1_CUT_ACTIVE
                    case A2bXform::Ts1Full:
                        ts1_public(m, (UINT_TYPE) 1 << (BITLENGTH - 2), g_a2b_xform_m[v], g_a2b_xform_sk[v]);
                        m = OP_SUB(g_a2b_xform_m[v], PROMOTE(kTs1A2bLow));
                        break;
#endif
                    case A2bXform::Ts1Shift:
                        m = ts1_shift_m0(m, g_a2b_xform_factor);  // the bit injection lifts it (ts1_lift_shift)
                        break;
#endif
                    case A2bXform::DCut:
                    case A2bXform::TeCut:
                        g_a2b_xform_m[v] = m;
                        m = OP_SHIFT_LOG_RIGHT<FRACTIONAL>(m);
                        break;
                    default:
                        break;
                }
            }
        });
}

// A2B_DCUT_ACTIVE: the inputs' m back (the bit injection takes the untruncated value)
template <typename Datatype, typename Share>
void a2b_xform_restore(sint_t<Additive_Share<Datatype, Share>>* val, int len)
{
    if constexpr (requires(Share& s) { s.raw_m(); })
        if (g_a2b_xform == A2bXform::DCut || g_a2b_xform == A2bXform::TeCut)
            for (int i = 0; i < len; i++)
            {
                auto* sh = val[i].get_share_pointer();
                for (int j = 0; j < BITLENGTH; j++) sh[j].raw_m() = g_a2b_xform_m[(uint64_t) i * BITLENGTH + j];
            }
}
#endif

// The MSB adders of a converted range: constructed (each retrieves its triples), then run level by level. b1 / b2:
// value i's operand bitsets (references: the narrow cut adders view slices FRACTIONAL.. of the full ones in place)
template <typename Adder, typename Share, typename LowAdder = void, typename B1, typename B2, typename S,
          typename LowBitset = int>
void run_msb_adders(B1 b1, B2 b2, S* msb, int len, LowBitset* l1 = nullptr, LowBitset* l2 = nullptr, S* lout = nullptr)
{
#if ADDITIONAL_RELU_THREADS > 0 && !(PPA4_MSB == 1 && ADDITIONAL_PPA_THREADS > 0)
    // constructed in parallel: each adder retrieves its triples and one random mask (stream cursors)
    struct AdderArray
    {
        Adder* p;
        int n = 0;
        explicit AdderArray(int len)
            : p(static_cast<Adder*>(::operator new[](sizeof(Adder) * len, std::align_val_t(alignof(Adder)))))
        {
        }
        ~AdderArray()
        {
            for (int i = 0; i < n; i++)
                p[i].~Adder();
            ::operator delete[](p, std::align_val_t(alignof(Adder)));
        }
        Adder& operator[](int i) { return p[i]; }
    } adders(len);
    stream_parallel_for<STREAM_PARALLEL_CTOR, true>(len, [&](int i) { new (&adders.p[i]) Adder(b1(i), b2(i), msb[i]); });
    adders.n = len;
#else
std::vector<Adder> adders;
    adders.reserve(len);
    for (int i = 0; i < len; i++)
    {
        adders.emplace_back(b1(i), b2(i), msb[i]);
    }
#endif
   
    // TE's low adders (see Ts1Range::te), constructed after the ReLU's and run in the same rounds
    struct NoLow
    {
        bool is_done() const { return true; }
        void step() {}
    };
    using Low = std::conditional_t<std::is_void_v<LowAdder>, NoLow, LowAdder>;
    std::vector<Low> low;
    if constexpr (!std::is_void_v<LowAdder>)
        if (l1)
        {
            low.reserve(len);
            for (int i = 0; i < len; i++) low.emplace_back(l1[i], l2[i], lout[i]);
        }
    auto low_done = [&]() { return low.empty() || low[0].is_done(); };
#if RESHARE_OPT == 1
Share::communicate(); // For resharings
#endif 
#if PPA4_MSB == 1 && ADDITIONAL_PPA_THREADS > 0
    while (!adders[0].is_done())
    {
        if(current_phase == PHASE_LIVE)
        // Spawn threads for compute_step (Live Phase has heavy computation)
       { 
        {
            std::vector<std::thread> threads;
            int chunk_size = (len + ADDITIONAL_PPA_THREADS - 1) / ADDITIONAL_PPA_THREADS;
            for (int t = 0; t < ADDITIONAL_PPA_THREADS && t * chunk_size < len; t++)
            {
                int start = t * chunk_size;
                int end = std::min(start + chunk_size, len);
                threads.emplace_back([&adders, start, end]() {
                    for (int i = start; i < end; i++)
                    {
                        adders[i].compute_step();
                    }
                });
            }
            for (auto& th : threads) th.join();
        }
    }
        else
        {
            for (int i = 0; i < len; i++)
            {
                adders[i].compute_step();
            }
        }
        // Aggregate step (single thread)
        for (int i = 0; i < len; i++)
        {
            adders[i].aggregate_step();
        }
        Share::communicate();
        for (int i = 0; i < len; i++)
        {
            adders[i].collect_step();
        }
    }
#else
    while (!adders[0].is_done() || !low_done())
    {
        const bool main_done = adders[0].is_done(), lows_done = low_done();
        stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) {
            if (!main_done)
                adders[i].step();
            if (!lows_done)
                low[i].step();
        });
        Share::communicate();
    }
#endif
#if !(ADDITIONAL_RELU_THREADS > 0 && !(PPA4_MSB == 1 && ADDITIONAL_PPA_THREADS > 0))
    adders.clear();
    adders.shrink_to_fit();
#endif
}

template <int bm, int bk, typename Datatype, typename Share>
void get_msb_range(sint_t<Additive_Share<Datatype, Share>>* val, XOR_Share<Datatype, Share>* msb, int len)
{
#if MASK_FORWARD_ACTIVE
    if (g_mask_pass)
        mask_pass_abort("an A2B outside a ReLU (the mask-only forward models ReLUs only)");
#endif
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using Bitset = sbitset_t<bk - bm, S>;
    using sint = sint_t<A>;
    Bitset* s1 = new Bitset[len];
    Bitset* s2 = new Bitset[len];
#if A2B_CONV_BAKE_ACTIVE
    // [c] was committed for mask lz of every A2B slot. A value its producer could not bake (see
    // g_msb_input_baked) is moved onto lz here; the delta is exchanged in preprocessing, so the online
    // phase stays free of A2B communication.
#if A2B_MASK_PASS_ACTIVE
    // The mask-only forward recorded each input's mask at its slot and [c] was built for it: nothing to move. In
    // preprocessing, check that the masks are what the mask forward saw (all but the last value group, whose
    // padding is not initialized).
#if TS1_FUSED_ACTIVE
    // TS1: the first ReLU's inputs move onto their committed masks (see g_a2b_rebase_now)
    if (g_a2b_rebase_now)
        for (int i = 0; i < len; i++)
        {
            auto* sh = val[i].get_share_pointer();
            for (int j = 0; j < BITLENGTH; j++)
                sh[j] = sh[j].rebase(a2b_bake_slot_mask<Datatype>((uint64_t) i * BITLENGTH + j, OP_SUB));
        }
    else
#endif
    if (current_phase == PHASE_PRE)
        for (int i = 0; i + 1 < len; i++)
        {
            auto* sh = val[i].get_share_pointer();
            for (int j = 0; j < BITLENGTH; j++)
                if (!equal_word(sh[j].get_mask(), a2b_bake_slot_mask<Datatype>((uint64_t) i * BITLENGTH + j, OP_SUB)))
                    mask_pass_abort("a ReLU input's mask differs from the mask-only forward's");
        }
#else
    // residual sums: the partner drew lz - (other addend), so the sum carries lz. P1 with truncation-image masks: only
    // if the other addend's mask is committed as well (a2b_residual_committed), otherwise P1 moves its part alone.
    const bool residual = msb_input_residual() && !msb_input_baked();
    const bool p1_moves = residual && !A2B_RESIDUAL_BAKE_P1 && !a2b_residual_committed(g_residual_k);
    const bool moved_by_producer = msb_input_baked() || (residual && (!p1_moves || PARTY == 0));
    if (current_phase == PHASE_INIT && residual && g_residual_k >= 0)
    {
        // the sum's A2B slots: the Boolean addition's slices counted so far (one per value and bit)
        ResidualSum& r = residual_sum(g_residual_k);
        if (r.slots == 0)
            r.slot_base = num_boolean_addition_triples;
        r.slots += (uint64_t) len * BITLENGTH;
    }
    // the bake's invariant: every such input's mask share is its slot's committed mask (all but the last value group,
    // whose padding is not initialized). Not checked for P1 with weights known in preprocessing, SecureML truncation
    // and dummy weights (MODELOWNER -1): its masks lie in the truncation's image, and the dummy biases give P1 a bias
    // mask, which the bake then cannot compensate (a real model owner's bias has no mask at P1).
    constexpr bool check_bake = !(PARTY == 1 && MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1 && TRUNC_DELAYED == 0 &&
                                  MODELOWNER == -1);
    if (check_bake && current_phase == PHASE_PRE && moved_by_producer)
        for (int i = 0; i + 1 < len; i++)
        {
            auto* sh = val[i].get_share_pointer();
            for (int j = 0; j < BITLENGTH; j++)
            {
                const Datatype want = a2b_bake_slot_mask<Datatype>((uint64_t) i * BITLENGTH + j, OP_SUB);
                const Datatype have = sh[j].get_share().get_mask();
                if (std::memcmp(&want, &have, sizeof(Datatype)) != 0)
                {
                    fprintf(stderr, "A2B_CONV_BAKE: a baked ReLU input's mask is not its committed mask (P%d, slot base %lu, "
                            "value %d of %d, residual %d, want %u, have %u)\n", (int) PARTY, (unsigned long) g_a2b_layer_base,
                            i * BITLENGTH + j, len * BITLENGTH, (int) residual, (unsigned) *(const UINT_TYPE*) &want,
                            (unsigned) *(const UINT_TYPE*) &have);
                    std::abort();
                }
            }
        }
    if (p1_moves)
        for (int i = 0; i < len; i++)
        {
            auto* sh = val[i].get_share_pointer();
            for (int j = 0; j < BITLENGTH; j++)
                sh[j] = sh[j].rebase_p1(a2b_bake_slot_mask<Datatype>((uint64_t) i * BITLENGTH + j, OP_SUB));
        }
    else if (!msb_input_baked() && !residual)
        for (int i = 0; i < len; i++)
        {
            auto* sh = val[i].get_share_pointer();
            for (int j = 0; j < BITLENGTH; j++)
                sh[j] = sh[j].rebase(a2b_bake_slot_mask<Datatype>((uint64_t) i * BITLENGTH + j, OP_SUB));
        }
#endif
#endif
#if TE_FUSED_ACTIVE
    // TE's low adders' inputs (see Ts1Range::te): (0, m mod 2^F) and (0, nu mod 2^F), slices l - F - 1 .. l - 1 of the
    // untransformed m and of the unshifted [c] (the slots the A2B below reads; not counted again in INIT), the top slice
    // replaced by a public 0
    constexpr int te_w = FRACTIONAL + 1, te_lo = BITLENGTH - FRACTIONAL - 1;
    using TeBitset = sbitset_t<te_w, S>;
    TeBitset* te1 = nullptr;
    TeBitset* te2 = nullptr;
    S* te_out = static_cast<S*>(g_te_low_out);
    if constexpr (bm == 0 && bk == BITLENGTH)
        if (te_out)
        {
            te1 = new TeBitset[len];
            te2 = new TeBitset[len];
            const uint64_t c_base = g_a2b_c_cursor;
            const int c_shift = g_a2b_c_shift;
            g_a2b_c_shift = 0;
            g_a2b_no_count = true;
            stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) {
                te1[i] = TeBitset::prepare_A2B_S1(te_lo, (S*) val[i].get_share_pointer());
                tl_a2b_c = (int64_t) (c_base + (uint64_t) i * BITLENGTH + te_lo);
                te2[i] = TeBitset::prepare_A2B_S2(te_lo, (S*) val[i].get_share_pointer());
                tl_a2b_c = -1;
                te1[i].complete_A2B_S1();
                te2[i].complete_A2B_S2();
                te1[i][0] = S(SET_ALL_ZERO());
                te2[i][0] = S(SET_ALL_ZERO());
            });
            g_a2b_no_count = false;
            g_a2b_c_shift = c_shift;
        }
#endif
#if TS1_FUSED_ACTIVE || A2B_DCUT_ACTIVE
    if (g_a2b_xform != A2bXform::None)
        a2b_xform_input(val, len);  // after the rebase: the masks are the committed ones the tuples were made for
#endif
#if A2B_ROUND_OPT_SIM == 0
    //Skip if we are simulating A2B with round optimization
#if A2B_CONV_BAKE_ACTIVE
    // [c] is addressed by value: value i reads the BITLENGTH slices from c_base + i * BITLENGTH on (tl_a2b_c), so the
    // values are prepared on the pool like the unbaked A2B's
    const uint64_t c_base = g_a2b_c_cursor;
    stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) {
        s1[i] = Bitset::prepare_A2B_S1(bm, (S*)val[i].get_share_pointer());
        tl_a2b_c = (int64_t) (c_base + (uint64_t) i * BITLENGTH);
        s2[i] = Bitset::prepare_A2B_S2(bm, (S*)val[i].get_share_pointer());
        tl_a2b_c = -1;
    });
    if (current_phase != PHASE_INIT)
        g_a2b_c_cursor = c_base + (uint64_t) len * BITLENGTH;
    Share::communicate();
    stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) {
        s1[i].complete_A2B_S1();
        s2[i].complete_A2B_S2();
    });
#else
    stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) {
        s1[i] = Bitset::prepare_A2B_S1(bm, (S*)val[i].get_share_pointer());
        s2[i] = Bitset::prepare_A2B_S2(bm, (S*)val[i].get_share_pointer());
    });
    Share::communicate();
    stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) {
        s1[i].complete_A2B_S1();
        s2[i].complete_A2B_S2();
    });
#endif
#endif
#if RESHARE_BAKE_ACTIVE && PPA4_MSB == 1
    // every group is prepared (only prepare_A2B_S1 reads the count): reset it here, serially, instead of
    // in the first adder's tuple retrieval, which would move it inside the parallel construction level
    g_a2b_s1_pending = 0;
#endif


#if CUT_FRAC_NARROW
    if constexpr (bm == 0 && bk == BITLENGTH)
        if (cut_frac_narrow_on(bm, bk))
        {
            // the cut at 64 bits: the narrow adder on slices FRACTIONAL..BITLENGTH-1 (see cut_frac_narrow_on)
            // views of slices FRACTIONAL.. of the full bitsets (sbitset_t is an array of shares)
            constexpr int w = BITLENGTH - FRACTIONAL;
            using NB = sbitset_t<w, S>;
            static_assert(sizeof(NB) == sizeof(S) * w && sizeof(Bitset) == sizeof(S) * BITLENGTH, "sbitset_t layout");
            auto n1 = [&](int i) -> NB& { return *reinterpret_cast<NB*>(s1[i].get_share_pointer() + FRACTIONAL); };
            auto n2 = [&](int i) -> NB& { return *reinterpret_cast<NB*>(s2[i].get_share_pointer() + FRACTIONAL); };
#if TE_FUSED_ACTIVE
            run_msb_adders<NARROW_ADDER_TYPE<w, S>, Share, TE_LOW_ADDER_TYPE<te_w, S>>(n1, n2, msb, len, te1, te2, te_out);
#else
            run_msb_adders<NARROW_ADDER_TYPE<w, S>, Share>(n1, n2, msb, len);
#endif
            delete[] s1;
            delete[] s2;
            s1 = s2 = nullptr;
        }
    if (s1)
#endif
    {
        auto f1 = [&](int i) -> Bitset& { return s1[i]; };
        auto f2 = [&](int i) -> Bitset& { return s2[i]; };
#if TE_FUSED_ACTIVE
        run_msb_adders<ADDER_TYPE<bk - bm, S>, Share, TE_LOW_ADDER_TYPE<te_w, S>>(f1, f2, msb, len, te1, te2, te_out);
#else
        run_msb_adders<ADDER_TYPE<bk - bm, S>, Share>(f1, f2, msb, len);
#endif
    }
    delete[] s1;
    delete[] s2;
#if TE_FUSED_ACTIVE
    delete[] te1;
    delete[] te2;
#endif
#if TS1_FUSED_ACTIVE || A2B_DCUT_ACTIVE
    a2b_xform_restore(val, len);
#endif
#if A2B_CONV_BAKE_ACTIVE
    // Advance the conv-mask layer base to the A2B group boundary. A2B packs BITLENGTH values per sint and
    // consumes BITLENGTH [c] slices even for a partial (< BITLENGTH) group, so the next layer's masks and
    // [c] must start at the same boundary. g_a2b_c_cursor is already at that boundary (it advanced one
    // slice per prepared A2B value). Identical in PRE and LIVE, so the msb-adder triples still match.
    g_a2b_layer_base = g_a2b_c_cursor;
#endif
}

template <int bm, int bk, typename Datatype, typename Share>
void A2B_range(sint_t<Additive_Share<Datatype, Share>>* val, sbitset_t<bk - bm, XOR_Share<Datatype, Share>>* y, int len)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using Bitset = sbitset_t<bk - bm, S>;
    using sint = sint_t<A>;
    Share::communicate();
    Bitset* s1 = new Bitset[len];
    Bitset* s2 = new Bitset[len];
    for (int i = 0; i < len; i++)
    {
        s1[i] = Bitset::prepare_A2B_S1(bm, (S*)val[i].get_share_pointer());
        s2[i] = Bitset::prepare_A2B_S2(bm, (S*)val[i].get_share_pointer());
    }
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
        s1[i].complete_A2B_S1();
        s2[i].complete_A2B_S2();
    }

    Share::communicate();

    std::vector<FULL_ADDER_TYPE<bk - bm, S>> adders;

    adders.reserve(len);
    for (int i = 0; i < len; i++)
    {
        adders.emplace_back(s1[i], s2[i], y[i]);
    }

    while (!adders[0].is_done())
    {
        for (int i = 0; i < len; i++)
        {
            adders[i].step();
        }
        /* std::cout << "Adder step ..." << std::endl; */
        Share::communicate();
    }
    delete[] s1;
    delete[] s2;
    adders.clear();
    adders.shrink_to_fit();
}

template <int bm, int bk, typename Datatype, typename Share>
void B2A_range(sbitset_t<bk - bm, XOR_Share<Datatype, Share>>* y, sint_t<Additive_Share<Datatype, Share>>* val, int len)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using Bitset = sbitset_t<bk - bm, S>;
    using sint = sint_t<A>;
    Bitset* random_mask = new Bitset[len];
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < bk - bm; j++)
        {
            random_mask[i][j].get_random_B2A();
        }
    }

    Bitset* z = new Bitset[len];
    std::vector<FULL_ADDER_TYPE<bk - bm, S>> adders2;

    adders2.reserve(len);
    for (int i = 0; i < len; i++)
    {
        adders2.emplace_back(y[i], random_mask[i], z[i]);
    }

    while (!adders2[0].is_done())
    {
        for (int i = 0; i < len; i++)
        {
            adders2[i].step();
        }
        Share::communicate();
    }
    adders2.clear();
    adders2.shrink_to_fit();
    delete[] y;
    for (int i = 0; i < len; i++)
    {
        sint::prepare_B2A(z[i].get_share_pointer(), random_mask[i].get_share_pointer(), val[i].get_share_pointer());
    }
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
        sint::complete_B2A(z[i].get_share_pointer(), val[i].get_share_pointer());
    }
#if PROTOCOL > 7  // 4PC protocols needs additional communication
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
        sint::complete_B2A2(z[i].get_share_pointer(), val[i].get_share_pointer());
    }
#endif
    delete[] z;
    delete[] random_mask;
}

// A2B_BAKE_MASK_PASS: while bit injection element i of a ReLU runs, its outputs take the committed masks of their
// slots (bi_output_mask); reset afterwards, also on the pool's workers
struct BiSlotScope
{
    explicit BiSlotScope(int i)
    {
#if MASK_FORWARD_ACTIVE
        tl_bi_slot = g_bi_base == UINT64_MAX ? -1 : (int64_t) (g_bi_base + (uint64_t) i * BITLENGTH);
#else
        (void) i;
#endif
    }
    ~BiSlotScope() { tl_bi_slot = -1; }
};

template <typename Datatype, typename Share>
void bit_injection_opt_range(XOR_Share<Datatype, Share>* y, sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
#if (FUSE_RELU_AVG == 1 || (TRUNC_DELAYED == 1 && BIT_INJECTION_TRUNC_SIM == 1)) && \
    (TRUNC_APPROACH == 4 || TRUNC_APPROACH == 0)
    // The optimized bit injection can fold a truncation into itself (prepare_opt_bit_injection_with_trunc computes
    // relu(val) * trunc_factor >> fractional_bits in one pass). Use it ONLY when there is something to scale /
    // truncate, decided at runtime so a plain ReLU stays cheap:
    //   - do_avg      : an average-pooling layer is fused into this ReLU (curr_denom > 1) -> multiply by 1/denom.
    //   - fold_delayed: a delayed conv/fc truncation is pending and we fold it here (BIT_INJECTION_TRUNC_SIM == 1).
    // Otherwise (plain ReLU, or a fused pool with denom == 1 and no pending truncation) use the normal,
    // non-truncating bit injection. The PRE phase generates the same triples either way, so this runtime choice is
    // safe (and curr_denom / delayed are synced between PRE and LIVE).
    bool do_avg = false;
#if FUSE_RELU_AVG == 1
    do_avg = (curr_denom > 1);
#endif
    bool fold_delayed = false;
#if TRUNC_DELAYED == 1 && BIT_INJECTION_TRUNC_SIM == 1
    fold_delayed = delayed;
#endif
    if (do_avg)
    {
        // Fold the 1/denom average-pool division into the truncation. If a delayed truncation is also pending the
        // input is still at scale 2^(2*FRACTIONAL), so truncate by 2*FRACTIONAL instead of FRACTIONAL.
        // AVG_RECIP_EXTRA_BITS: the reciprocal gets that many more fractional bits (truncated away again): with
        // FRACTIONAL = 5 alone, 1/9 (the ResNet stem's 3x3 pool) becomes 4/32 = 0.125, a 12.5% scaling error.
        auto reciprocal = FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(
            1 / FLOATTYPE(curr_denom), FRACTIONAL + AVG_RECIP_EXTRA_BITS);
        const int fb = (fold_delayed ? 2 * FRACTIONAL : FRACTIONAL) + AVG_RECIP_EXTRA_BITS;
        stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) {
            BiSlotScope bi(i);
            y[i].prepare_opt_bit_injection_with_trunc(val[i].get_share_pointer(), val[i].get_share_pointer(),
                                                      PROMOTE(reciprocal), fb);
        });
    }
    else if (fold_delayed)
    {
        // Pure delayed truncation (no avg pool, denom == 1): just shift right by FRACTIONAL. Use multiplier 1
        // rather than the fixed-point "1.0" (== 2^FRACTIONAL) so we don't scale up by 2^FRACTIONAL first (which
        // would risk overflow on larger activations) - mathematically identical, but safer.
        stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) {
            BiSlotScope bi(i);
            y[i].prepare_opt_bit_injection_with_trunc(val[i].get_share_pointer(), val[i].get_share_pointer(),
                                                      PROMOTE(UINT_TYPE(1)), FRACTIONAL);
        });
    }
    else
    {
        stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) {
            BiSlotScope bi(i);
            y[i].prepare_opt_bit_injection(val[i].get_share_pointer(), val[i].get_share_pointer());
        });
    }
#if TRUNC_DELAYED == 1 && BIT_INJECTION_TRUNC_SIM == 1
    if (fold_delayed)
        delayed = false;  // consumed here, so the caller's trailing `if (delayed)` must not truncate again
#endif
#else
    for (int i = 0; i < len; i++)
    {
        BiSlotScope bi(i);
        y[i].prepare_opt_bit_injection(val[i].get_share_pointer(), val[i].get_share_pointer());
    }
#endif
    Share::communicate();
    stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) { val[i].complete_opt_bit_injection(); });
}

#if TS1_FUSED_ACTIVE
// TS1 fused into the ReLU (2PC, see g_ts1_la): val holds the delayed ReLU inputs, which become DReLU * trunc(val);
// ts1 is the compact index of val[0]'s first value. The truncated values' public parts and sK: lift_tp >= 0 (shifted
// design) by ts1_lift_shift from val's m (M0), else M, sk by value (full design with the cut) or by ts1_public from
// val's m. fb > 0: the product times trunc_factor is truncated probabilistically by fb (a pooling fused into the ReLU,
// TS_Mix).
template <typename Datatype, typename Share>
void bit_injection_ts1_range(XOR_Share<Datatype, Share>* y,
                             sint_t<Additive_Share<Datatype, Share>>* val,
                             const int len,
                             uint64_t ts1,
                             const Datatype* M,
                             const Datatype* sk,
                             int lift_tp,
                             Datatype trunc_factor,
                             int fb,
                             const XOR_Share<Datatype, Share>* te_c = nullptr)
{
    stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) {
        BiSlotScope bi(i);
        const uint64_t o = (uint64_t) i * BITLENGTH;
        y[i].prepare_opt_bit_injection_ts1(val[i].get_share_pointer(), val[i].get_share_pointer(), ts1 + o,
                                           M ? M + o : nullptr, sk ? sk + o : nullptr, lift_tp, trunc_factor, fb,
                                           te_c ? static_cast<const Share*>(&te_c[i]) : nullptr);
    });
    Share::communicate();
    stream_parallel_for<STREAM_PARALLEL_RELU, true>(len, [&](int i) { val[i].complete_opt_bit_injection(); });
}
#endif

#if MASK_FORWARD_ACTIVE
// The mask-only forward (SimpleNN::evaluate, preprocessing pass, before the real forward, on a copy of the input):
// forward() with every ReLU outputting its committed masks. UC3 (A2B_BAKE_MASK_PASS): the ReLUs record their input
// masks, all other layers run their preprocessing code (public weights draw and write nothing but the truncations'
// masks), then the Boolean addition runs on the recorded masks. Secret weights (CHEETAH_CONV_EARLY): the convs record
// their triples' inputs, conv, FC and BatchNorm layers stop there, then the conv triples start and the OT phase runs.
// The random generators are restored, pre-sends are dropped, the triple-type index is restored, and every other
// preprocessing stream must not have moved. The real forward then starts from the same state.
template <typename F>
void mask_forward(F&& forward)
{
    constexpr int links = num_players * player_multiplier;
    AES_TYPE counters[links];
    uint64_t generated[links];
    for (int l = 0; l < links; l++) counters[l] = aes_counter[l], generated[l] = num_generated[l];
#if TRUNC_DELAYED == 1
    const bool delayed0 = delayed;
#endif
#if FUSE_RELU_AVG == 1
    const auto denom0 = curr_denom;
#endif
#if STREAM_PARALLEL_PRE_ACTIVE
    const auto before = stream_parallel::pre_flat(stream_parallel::pre_snapshot());
#endif
#if ADDITIONAL_RELU_THREADS > 0
    const auto rnd0 = rnd_calls_self;
#endif
    const uint64_t send0 = send_count_pre[PNEXT];
    // the truncations of the pooling layers record a triple type per value (for the online phase's bookkeeping):
    // an append-only stream the real forward writes again from the same position
    const auto types0 = triple_type_index;
#if A2B_MASK_PASS_ACTIVE
    g_a2b_layer_base = 0;
#endif
#if CHEETAH_CONV_EARLY_ACTIVE
    g_early_conv_index = 0;
#endif
    g_relu_base = 0;
    g_lin_counter = 0;
    g_mask_pass = true;
    forward();
    g_mask_pass = false;
    if (g_relu_base != g_relu_out.size())
        mask_pass_abort("did not reach every ReLU");
    g_relu_base = 0;
    g_lin_counter = 0;  // the real forward draws the same truncation masks
#if A2B_MASK_PASS_ACTIVE
    if (g_a2b_layer_base != g_a2b_lz.size())
        mask_pass_abort("did not reach every A2B slot");
#endif
#if CHEETAH_CONV_EARLY_ACTIVE
    if (g_early_conv_index != conv_triple_params.size())
        mask_pass_abort("did not reach every conv");
#endif
    for (int l = 0; l < links; l++) aes_counter[l] = counters[l], num_generated[l] = generated[l];
#if TRUNC_DELAYED == 1
    delayed = delayed0;
#endif
#if FUSE_RELU_AVG == 1
    curr_denom = denom0;
#endif
#if ADDITIONAL_RELU_THREADS > 0
    rnd_calls_self = rnd0;
#endif
    for (size_t r = 0; r < types0.size() && r < triple_type_index.size(); r++) triple_type_index[r] = types0[r];
#if STREAM_PARALLEL_PRE_ACTIVE
    if (const auto now = stream_parallel::pre_flat(stream_parallel::pre_snapshot()); now != before)  // every stream where it was
    {
        for (size_t k = 0; k < now.size(); k++)
            if (now[k] != before[k])
                fprintf(stderr, "mask-only forward: preprocessing stream %zu moved from %ld to %ld\n", k, (long) before[k], (long) now[k]);
        mask_pass_abort("wrote preprocessing material");
    }
#endif
    if (send_count_pre[PNEXT] != send0)
        mask_pass_abort("wrote preprocessing material");
#if A2B_MASK_PASS_ACTIVE
    g_mask_pass_hook();
    g_a2b_layer_base = 0;
    g_a2b_c_cursor = 0;
#endif
#if CHEETAH_CONV_EARLY_ACTIVE
    g_early_ot_hook();  // starts the conv triples, then runs the OT phase
#endif
}
#endif

template <typename Share, typename Datatype>
void bit2A_range(XOR_Share<Datatype, Share>* bit_val, int len, sint_t<Additive_Share<Datatype, Share>>* output)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using sint = sint_t<A>;
    for (int i = 0; i < len; i++)
    {
        bit_val[i].prepare_bit2a(output[i].get_share_pointer());
    }
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
        output[i].complete_bit2a();
    }
}

template <typename Share, typename Datatype>
void bitinj_range(XOR_Share<Datatype, Share>* bit_val, int len, sint_t<Additive_Share<Datatype, Share>>* output)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using sint = sint_t<A>;
    sint* t1 = new sint[len];
    sint* t2 = new sint[len];
    for (int i = 0; i < len; i++)
    {
        bit_val[i].prepare_bit_injection_S1(t1[i].get_share_pointer());
        bit_val[i].prepare_bit_injection_S2(t2[i].get_share_pointer());
    }
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
        t1[i].complete_bit_injection_S1();
        t2[i].complete_bit_injection_S2();
    }
    for (int i = 0; i < len; i++)
    {
        output[i].prepare_XOR(t1[i], t2[i]);
    }
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
        output[i].complete_XOR(t1[i], t2[i]);
    }
    delete[] t1;
    delete[] t2;
}

template <int rm = 0, int rk = BITLENGTH, typename Share, typename Datatype, typename FUNC_OP>
static void pack_additive(const Additive_Share<Datatype, Share>* input,
                          Additive_Share<Datatype, Share>* output,
                          const int len,
                          FUNC_OP op)
{
    using A = Additive_Share<Datatype, Share>;
    using sint = sint_t<A>;
    int m = len;
    sint* tmp = new sint[(m - 1) / BITLENGTH + 1];
    sint* tmp_output = new sint[(m - 1) / BITLENGTH + 1];
    int counter = 0;
    while (m > (BITLENGTH - 1))
    {
        tmp[counter] = sint::load_shares(input + counter * BITLENGTH);
        counter++;
        m -= BITLENGTH;
    }
    if (m > 0)
    {
        tmp[counter] = sint::load_shares(m, input + counter * BITLENGTH);
        counter++;
    }
    op(tmp, tmp_output, counter);
    counter = 0;
    m = len;
    while (m > (BITLENGTH - 1))
    {
        for (int j = 0; j < BITLENGTH; j++)
        {
            output[counter * BITLENGTH + j] = tmp_output[counter].get_share(j);
        }
        counter++;
        m -= BITLENGTH;
    }
    if (m > 0)
    {
        for (int j = 0; j < m; j++)
        {
            output[counter * BITLENGTH + j] = tmp_output[counter].get_share_pointer()[j];
        }
    }
    delete[] tmp;
    delete[] tmp_output;
}

template <int rm = 0, int rk = BITLENGTH, typename Share, typename Datatype, typename FUNC_OP>
static void pack_additive_inplace(const Additive_Share<Datatype, Share>* input,
                                  Additive_Share<Datatype, Share>* output,
                                  const int len,
                                  FUNC_OP op)
{
    using A = Additive_Share<Datatype, Share>;
    using sint = sint_t<A>;
    int m = len;
    sint* tmp = new sint[(m - 1) / BITLENGTH + 1];
    int counter = 0;
    while (m > (BITLENGTH - 1))
    {
        tmp[counter] = sint::load_shares(input + counter * BITLENGTH);
        counter++;
        m -= BITLENGTH;
    }
    if (m > 0)
    {
        tmp[counter] = sint::load_shares(m, input + counter * BITLENGTH);
        counter++;
    }
    op(tmp, counter);
    counter = 0;
    m = len;
    while (m > (BITLENGTH - 1))
    {
        for (int j = 0; j < BITLENGTH; j++)
        {
            output[counter * BITLENGTH + j] = tmp[counter].get_share(j);
        }
        counter++;
        m -= BITLENGTH;
    }
    if (m > 0)
    {
        for (int j = 0; j < m; j++)
        {
            output[counter * BITLENGTH + j] = tmp[counter].get_share_pointer()[j];
        }
    }
    delete[] tmp;
}

template <int rm = 0, int rk = BITLENGTH, typename Share, typename Datatype, typename FUNC_OP>
static void pack_additive_inplace(Additive_Share<Datatype, Share>* val, const int len, FUNC_OP op)
{
    using sint = sint_t<Additive_Share<Datatype, Share>>;
    int m = len;
    sint* tmp = new sint[(m - 1) / BITLENGTH + 1];
    int counter = 0;
    while (m > BITLENGTH - 1)
    {
        tmp[counter] = sint::load_shares(val + counter * BITLENGTH);
        counter++;
        m -= BITLENGTH;
    }
    if (m > 0)
    {
        tmp[counter] = sint::load_shares(m, val + counter * BITLENGTH);
        counter++;
    }
    /* RELU_range_in_place<rm,rk,Share>(tmp, counter); */
    op(tmp, counter);
    counter = 0;
    m = len;
    while (m > BITLENGTH - 1)
    {
        for (int j = 0; j < BITLENGTH; j++)
        {
            val[counter * BITLENGTH + j] = tmp[counter].get_share(j);
        }
        counter++;
        m -= BITLENGTH;
    }
    if (m > 0)
    {
        for (int j = 0; j < m; j++)
        {
            val[counter * BITLENGTH + j] = tmp[counter].get_share_pointer()[j];
        }
    }
    delete[] tmp;
}

template <int rm = 0, int rk = BITLENGTH, typename Share, typename Datatype, typename FUNC_OP>
static void pack_additive_inplace(Additive_Share<Datatype, Share>* val,
                                  const int len,
                                  const int fractiona_bits,
                                  FUNC_OP op)
{
    using sint = sint_t<Additive_Share<Datatype, Share>>;
    int m = len;
    sint* tmp = new sint[(m - 1) / BITLENGTH + 1];
    int counter = 0;
    while (m > BITLENGTH - 1)
    {
        tmp[counter] = sint::load_shares(val + counter * BITLENGTH);
        counter++;
        m -= BITLENGTH;
    }
    if (m > 0)
    {
        tmp[counter] = sint::load_shares(m, val + counter * BITLENGTH);
        counter++;
    }
    /* RELU_range_in_place<rm,rk,Share>(tmp, counter); */
    op(tmp, counter, fractiona_bits);
    counter = 0;
    m = len;
    while (m > BITLENGTH - 1)
    {
        for (int j = 0; j < BITLENGTH; j++)
        {
            val[counter * BITLENGTH + j] = tmp[counter].get_share(j);
        }
        counter++;
        m -= BITLENGTH;
    }
    if (m > 0)
    {
        for (int j = 0; j < m; j++)
        {
            val[counter * BITLENGTH + j] = tmp[counter].get_share_pointer()[j];
        }
    }
    delete[] tmp;
}
