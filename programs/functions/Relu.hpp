#pragma once
#include "share_conversion.hpp"
#if TRUNC_APPROACH > 0
#include "exact_truncation.hpp"
#endif
#include "prob_truncation.hpp"
/* #include "boolean_adder_bandwidth.hpp" */

#if OPTIMIZED_BIT_INJECTION_RELU == 1
#define RELU_range_in_place RELU_range_in_place_opt
#else
#define RELU_range_in_place RELU_range_in_place_optB2A
#endif

#if TTP_PROTOCOL == 0 || SIMULATE_MPC_FUNCTIONS == 1

#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3

template <int bm, int bk, typename Share, typename Datatype>
void RELU_range_in_place_exact(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<bk - bm, S>;
    using sint = sint_t<A>;

    Bitset* y = new Bitset[len];
    A2B_range<bm, bk, Datatype, Share>(val, y, len);

    for (int i = 0; i < len; i++)
    {
        auto msb = ~y[i][0];
        for (int j = bm; j < bk; j++)
        {
            y[i][j] = y[i][j] & msb;
        }
    }
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
        for (int j = bm; j < bk; j++)
            y[i][j].complete_and();
    }
    Share::communicate();

#if TRUNC_DELAYED == 1
    if (delayed)
    {
        for (int i = 0; i < len; i++)
        {
            for (int j = bk - 1; j >= FRACTIONAL; j--)
            {
                y[i][j] = y[i][j - FRACTIONAL];  // shift right
            }
            for (int j = bm; j < FRACTIONAL; j++)
            {
                y[i][j] = SET_ALL_ZERO();  // set most significant bits to zero
            }
        }
    }
#endif
    B2A_range<bm, bk, Datatype, Share>(y, val, len);
}
#endif

template <int m, int k, typename Share, typename Datatype>
void RELU_range_in_place_opt(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<k - m, S>;
    using sint = sint_t<A>;

    Share::communicate();

#if TRUNC_DELAYED == 1 && BIT_INJECTION_TRUNC_SIM == 1 && FUSE_RELU_AVG == 1 && PUBLIC_WEIGHTS == 0 && \
    (TRUNC_APPROACH == 0 || TRUNC_APPROACH == 4)
    // When BOTH a linear delayed truncation (from the preceding conv/fc) and an average-pool division (denom > 1)
    // fall on this ReLU, handle the linear delayed truncation up front here (as conv/avgpool do under if(delayed))
    // and let the bit injection fold ONLY the avg division. This splits one 2*FRACTIONAL local truncation into two
    // FRACTIONAL ones, which is more robust (lower SecureML wrap probability).
    // GATED to PUBLIC_WEIGHTS == 0: for public weights trunc_pr_in_place on the RAW conv output wraps systematically
    // (same issue as prepare_mult_public_fixed), so public weights instead fold both into the re-masking bit
    // injection (which works), see bit_injection_opt_range.
    if (delayed && curr_denom > 1)
    {
        trunc_pr_in_place(val, len);
        delayed = false;  // consumed: bit_injection_opt_range now sees fold_delayed == false (avg division only)
    }
#endif

    S* y = new S[len];
#if (CUT_FRAC_ELIGIBLE && TRUNC_DELAYED == 0) || CUT_FRAC_ELIGIBLE_GENERIC
    g_cut_frac_active = true;
#elif CUT_FRAC_ELIGIBLE  // TD=1 (TS1, A2B_DELAYED_CUT): an input truncated before (by a pooling) is at scale 2^F too
    g_cut_frac_active = !delayed;
#endif
#if A2B_DCUT_ACTIVE
    // A2B_DELAYED_CUT: a delayed input's A2B converts the locally truncated value (m >> F, and the bake adds the mask
    // shares >> F), with the cut; the bit injection then takes the untruncated value as before
    std::unique_ptr<Datatype[]> saved_m(delayed && current_phase == PHASE_LIVE ? new Datatype[(size_t) len * BITLENGTH]
                                                                                : nullptr);
    if (delayed)
    {
        if (current_phase == PHASE_INIT)
            a2b_record_shift(num_boolean_addition_triples, (uint64_t) len * BITLENGTH, 1);
        g_a2b_xform = current_phase == PHASE_LIVE ? A2bXform::DCut : A2bXform::None;
        g_a2b_xform_m = saved_m.get();
        g_cut_frac_active = true;
    }
#endif
#if A2B_DCUT_SHARE_ACTIVE
    // A2B_DELAYED_CUT without the bake: a delayed input's A2B converts the parties' shares shifted by F (P0's
    // m - l_0, P1's -l_1), whose sum is trunc(z) - 0 or 1, with the cut; the bit injection takes z as before. Not
    // where RESHARE_OPT_SIM baked P1's reshare material into the unshifted share.
    if (delayed && !reshare_sim_on())
    {
        g_a2b_share_shift = true;
        g_cut_frac_active = true;
    }
#endif
    get_msb_range<m, k, Datatype, Share>(val, y, len);
    g_cut_frac_active = false;
#if A2B_DCUT_ACTIVE
    g_a2b_xform = A2bXform::None;
    g_a2b_xform_m = nullptr;
#endif
#if A2B_DCUT_SHARE_ACTIVE
    g_a2b_share_shift = false;
#endif

    for (int i = 0; i < len; i++)
    {
        y[i] = ~y[i];
    }

    bit_injection_opt_range<Datatype, Share>(y, val, len);

    delete[] y;

    Share::communicate();
#if TRUNC_DELAYED == 1
    if (delayed)
    {
#if TRUNC_APPROACH == 0 && BIT_INJECTION_TRUNC_SIM == 0
        trunc_pr_in_place(val, len);
#elif TRUNC_APPROACH == 1 || TRUNC_APPROACH == 4
        print_online("Shouldn't be here");
        trunc_2k_in_place(val, len, true);
#elif TRUNC_APPROACH == 2
        print_online("Shouldn't be here");
        trunc_exact_in_place<Datatype, Share, void>(val, len);
#elif TRUNC_APPROACH == 3
        trunc_exact_opt_in_place<Datatype, Share, void>(val, len);
        print_online("Shouldn't be here");
#endif
    }
#endif

    /* } */
}

#if TS1_FUSED_ACTIVE
// TRUNC_APPROACH 1 / 4 (2PC): a delayed input's truncation happens in the bit injection (see g_ts1_la), from the
// A2B slots' TS1 tuples. A pooling fused into the ReLU (FUSE_RELU_AVG): TRUNC_APPROACH 1 folds 1/denom into TS1 (the
// shifted design's factor and lift truncation), 4 truncates the product probabilistically (TS_Mix).
template <int m, int k, typename Share, typename Datatype>
void RELU_range_in_place_ts1(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    static_assert(m == 0 && k == BITLENGTH, "TS1: full-width ReLUs only");
    using S = XOR_Share<DATATYPE, Share>;
    Share::communicate();
    const uint64_t slots = (uint64_t) len * BITLENGTH;  // the A2B slots get_msb_range takes
    UINT_TYPE factor = 1;
    int tp = 0;
    Datatype trunc_factor = SET_ALL_ZERO();
    int fb = 0;
#if FUSE_RELU_AVG == 1
    if (curr_denom > 1)
    {
        const UINT_TYPE reciprocal = FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(
            1 / FLOATTYPE(curr_denom), FRACTIONAL + AVG_RECIP_EXTRA_BITS);
#if TS1_FOLD_POOL
#if !TS1_CUT_ACTIVE
        fprintf(stderr, "TRUNC_APPROACH 1 (2PC): a pooling fused into a ReLU needs the shifted TS1 design (the cut)\n");
        std::abort();
#endif
        factor = reciprocal;
        tp = (delayed ? FRACTIONAL : 0) + AVG_RECIP_EXTRA_BITS;  // the lift's part of the truncation (F is local)
#else
        trunc_factor = PROMOTE(reciprocal);
        fb = FRACTIONAL + AVG_RECIP_EXTRA_BITS;  // the value is at scale 2^FRACTIONAL already
#endif
    }
#endif
#if TS1_CUT_ACTIVE
    const bool shift = TS1_LOW_CARRY == 0 || factor != 1;
#else
    const bool shift = false;
#endif
    uint64_t ts1 = 0;
    if (current_phase == PHASE_INIT)
    {
        ts1_record_range(num_boolean_addition_triples, slots, shift, factor, tp);
        if (shift)
            a2b_record_shift(num_boolean_addition_triples, slots, factor);
        else if (TS1_CUT_ACTIVE)
            g_a2b_full_width = true;  // the full design reads the carries up to the top
    }
    else
    {
        const Ts1Range& r = ts1_range(g_a2b_c_cursor, slots);
        if (r.shift != shift || r.factor != factor || r.tp != tp)
        {
            fprintf(stderr, "TS1: a ReLU's design differs from the INIT pass's\n");
            std::abort();
        }
        ts1 = r.compact_base;
    }
    S* y = new S[len];
#if TS1_CUT_ACTIVE
    // DReLU of the truncated value, whose top FRACTIONAL bits are sign extension: the A2B takes its public part and
    // the bake's [c] (shifted design: of the shifted shares; full design: shifted by FRACTIONAL slices), with the cut
    // (the shifted design's lift needs only M0, the full design's M and sK are kept by value)
    std::vector<Datatype> M(current_phase == PHASE_LIVE && !shift ? slots : 0), sk(M.size());
    g_a2b_xform = current_phase == PHASE_LIVE ? (shift ? A2bXform::Ts1Shift : A2bXform::Ts1Full) : A2bXform::None;
    g_a2b_xform_m = M.data();
    g_a2b_xform_sk = sk.data();
    g_a2b_xform_factor = factor;
    g_a2b_c_shift = shift ? 0 : FRACTIONAL;
    g_cut_frac_active = true;
    get_msb_range<m, k, Datatype, Share>(val, y, len);
    g_cut_frac_active = false;
    g_a2b_c_shift = 0;
    g_a2b_xform = A2bXform::None;
    g_a2b_xform_m = g_a2b_xform_sk = nullptr;
    for (int i = 0; i < len; i++)
        y[i] = ~y[i];
    bit_injection_ts1_range<Datatype, Share>(y, val, len, ts1, M.empty() ? nullptr : M.data(),
                                             sk.empty() ? nullptr : sk.data(), shift ? tp : -1, trunc_factor, fb);
#else
    get_msb_range<m, k, Datatype, Share>(val, y, len);
    for (int i = 0; i < len; i++)
        y[i] = ~y[i];
    bit_injection_ts1_range<Datatype, Share>(y, val, len, ts1, nullptr, nullptr, -1, trunc_factor, fb);
#endif
    delete[] y;
    Share::communicate();
}
#endif

template <int m, int k, typename Share, typename Datatype>
void RELU_range_in_place_optB2A(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<k - m, S>;
    using sint = sint_t<A>;

    S* y = new S[len];
#if (CUT_FRAC_ELIGIBLE && TRUNC_DELAYED == 0) || CUT_FRAC_ELIGIBLE_GENERIC
    g_cut_frac_active = true;
#elif CUT_FRAC_ELIGIBLE  // TD=1 (TS1, A2B_DELAYED_CUT): an input truncated before (by a pooling) is at scale 2^F too
    g_cut_frac_active = !delayed;
#endif
    get_msb_range<m, k, Datatype, Share>(val, y, len);
    g_cut_frac_active = false;
    for (int i = 0; i < len; i++)
    {
        y[i] = ~y[i];
    }
    /* if(current_phase == 1) */
    /*     std::cout << "Bit inj ..." << std::endl; */

    sint* result = new sint[len];
    for (int i = 0; i < len; i++)
    {
        y[i].prepare_bit2a(result[i].get_share_pointer());
    }
    delete[] y;
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
        result[i].complete_bit2a();
    }

    Share::communicate();

#if TRUNC_APPROACH == 0 && TRUNC_DELAYED == 1
    if (delayed)
    {
        for (int i = 0; i < len; i++)
        {
            val[i] = result[i].prepare_dot(val[i]);
            val[i].mask_and_send_dot();
        }
    }
    else
    {
        for (int i = 0; i < len; i++)
        {
            val[i] = result[i].prepare_dot(val[i]);
            val[i].mask_and_send_dot_without_trunc();
        }
    }
#else
    for (int i = 0; i < len; i++)
    {
        val[i] = result[i].prepare_dot(val[i]);
        val[i].mask_and_send_dot_without_trunc();
    }
#endif
    delete[] result;
    Share::communicate();
#if TRUNC_APPROACH == 0 && TRUNC_DELAYED == 1
    if (delayed)
        for (int i = 0; i < len; i++)
            val[i].complete_mult();
    else
        for (int i = 0; i < len; i++)
            val[i].complete_mult_without_trunc();
#else
    for (int i = 0; i < len; i++)
    {
        val[i].complete_mult_without_trunc();
    }
#endif
    /* val[i] -= sint(1); // To counter the +1 in TRUNC */

    Share::communicate();

#if TRUNC_DELAYED == 1 && TRUNC_APPROACH > 0
    if (delayed)
    {
#if TRUNC_APPROACH == 1 || TRUNC_APPROACH == 4
        trunc_2k_in_place(val, len, false);
#elif TRUNC_APPROACH == 2
        print_online("Shouldn't be here");
        trunc_exact_in_place<void>(val, len);
#elif TRUNC_APPROACH == 3
        print_online("Shouldn't be here");
        trunc_exact_opt_in_place<Datatype, Share, void>(val, len);
#endif
    }

#endif
}

#else

template <int m, int k, typename Share, typename Datatype>
void RELU_range_in_place(sint_t<Additive_Share<Datatype, Share>>* val, int len)
{
    for (int i = 0; i < len; i++)
        val[i] = val[i].relu();
}

template <int m, int k, typename Share, typename Datatype>
void RELU_range_in_place(Additive_Share<Datatype, Share>* val, int len)
{
    for (int i = 0; i < len; i++)
        val[i] = val[i].relu();
}

template <int m, int k, typename Share, typename Datatype>
void RELU_range_in_place_opt(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    for (int i = 0; i < len; i++)
        val[i] = val[i].relu();
}

template <int m, int k, typename Share, typename Datatype>
void RELU_range_in_place_exact(Additive_Share<Datatype, Share>* val, const int len)
{
    for (int i = 0; i < len; i++)
        val[i] = val[i].relu();
}

#endif

#if PRINT_OUTPUT_HASH == 1
// Debugging (env DEBUG_RELU0): reveal and print the first values of the first ReLU's input and output (every phase
// reveals, so the passes stay in step; the online phase prints)
template <typename A>
inline void debug_relu0_reveal(const A* v, int n, const char* what, int stride)
{
    for (int i = 0; i < n; i++) v[i * stride].prepare_reveal_to_all();
    A::communicate();
    std::string s;
    for (int i = 0; i < n; i++)
    {
        auto r = v[i * stride].complete_reveal_to_all();
        alignas(sizeof(DATATYPE)) UINT_TYPE o[DATTYPE / BITLENGTH];
        unorthogonalize_arithmetic(&r, o, 1);
        s += std::to_string((long long) (INT_TYPE) o[0]) + " ";
    }
    if (current_phase == PHASE_LIVE)
        print("DEBUG_RELU0 %s: %s\n", what, s.c_str());
}
#endif

template <int rm = 0, int rk = BITLENGTH, typename Share, typename Datatype>
static void RELU(const Additive_Share<Datatype, Share>* begin,
                 const Additive_Share<Datatype, Share>* end,
                 Additive_Share<Datatype, Share>* output)
{
    const int len = end - begin;
#if PRINT_OUTPUT_HASH == 1
    static int dbg_phase = -1;
#if MASK_FORWARD_ACTIVE
    const bool dbg = getenv("DEBUG_RELU0") && dbg_phase != current_phase && len >= 16 && !g_mask_pass;
#else
    const bool dbg = getenv("DEBUG_RELU0") && dbg_phase != current_phase && len >= 16;
#endif
    if (dbg)
    {
        dbg_phase = current_phase;
        debug_relu0_reveal(begin, 16, "in", len / 16);
    }
    struct DbgOut
    {
        bool on;
        Additive_Share<Datatype, Share>* o;
        int stride;
        ~DbgOut()
        {
            if (on)
                debug_relu0_reveal(o, 16, "out", stride);
        }
    } dbg_out{dbg, output, len / 16};
#endif
#if MASK_FORWARD_ACTIVE
    // this ReLU's committed output masks: slots relu_base.. (BITLENGTH per packed sint), counted in the INIT pass
    const uint64_t relu_slots = (uint64_t) ((len + BITLENGTH - 1) / BITLENGTH) * BITLENGTH;
    if (current_phase == PHASE_INIT)
        g_relu_slots += relu_slots;
    const uint64_t relu_base = g_relu_base;
    g_relu_base += relu_slots;
#if A2B_CONV_BAKE_ACTIVE
    if (current_phase == PHASE_INIT && g_relu_identity_k >= 0)
        residual_sum(g_relu_identity_k).relu_base = relu_base;  // a residual sum's other addend (a2b_residual_other_mask)
#endif
    if (g_mask_pass)
    {
#if A2B_MASK_PASS_ACTIVE
        a2b_mask_pass_record<Datatype, Share>(begin, len, relu_base == 0);
#endif
        mask_pass_relu_outputs<Datatype, Share>(len, output, relu_base);
#if TRUNC_DELAYED == 1
        delayed = false;
#endif
#if TRUNC_APPROACH > 0
        all_positive = true;
#endif
        return;
    }
    if (current_phase != PHASE_INIT)
        g_bi_base = relu_base;  // the bit injection gives the outputs these masks
#if TS1_FUSED_ACTIVE && A2B_MASK_PASS_ACTIVE
    g_a2b_rebase_now = relu_base == 0;  // see g_a2b_rebase_now
    struct RebaseOff
    {
        ~RebaseOff() { g_a2b_rebase_now = false; }
    } rebase_off;
#endif
#endif
#if TRUNC_DELAYED == 1 && TRUNC_APPROACH > 0
    if (delayed)
    {
#if TRUNC_APPROACH == 2
        pack_additive_inplace<rm, rk>(begin, output, len, RELU_range_in_place_exact<rm, rk, Share, Datatype>);
#elif TS1_FUSED_ACTIVE
        pack_additive_inplace<rm, rk>(begin, output, len, RELU_range_in_place_ts1<rm, rk, Share, Datatype>);
#else
        isReLU = true;
        std::copy(begin, end, output);
        trunc_exact_opt_in_place(output, len, true);
        isReLU = false;
#endif
    }
#if TS1_FUSED_ACTIVE && TS1_FOLD_POOL && FUSE_RELU_AVG == 1
    else if (curr_denom > 1)  // a pooling fused into a ReLU on a truncated value: TS1 divides
        pack_additive_inplace<rm, rk>(begin, output, len, RELU_range_in_place_ts1<rm, rk, Share, Datatype>);
#endif
    else
        pack_additive_inplace<rm, rk>(begin, output, len, RELU_range_in_place<rm, rk, Share, Datatype>);
#else
        pack_additive_inplace<rm, rk>(begin, output, len, RELU_range_in_place<rm, rk, Share, Datatype>);
#endif

#if MASK_FORWARD_ACTIVE
    g_bi_base = UINT64_MAX;
#endif
#if TRUNC_DELAYED == 1
    delayed = false;
#endif

#if TRUNC_APPROACH > 0
    all_positive = true;
#endif
}

template <int m = 0, int k = BITLENGTH, typename Share, typename Datatype>
static void RELU(const sint_t<Additive_Share<Datatype, Share>>* begin,
                 const sint_t<Additive_Share<Datatype, Share>>* end,
                 sint_t<Additive_Share<Datatype, Share>>* output)
{
    std::copy(begin, end, output);
    int len = end - begin;
#if TRUNC_DELAYED == 1 && TRUNC_APPROACH > 0
    if (delayed)
    {
#if TRUNC_APPROACH == 2
        RELU_range_in_place_exact<m, k, Share, Datatype>(output, len);
#elif TS1_FUSED_ACTIVE
        RELU_range_in_place_ts1<m, k, Share, Datatype>(output, len);
#else
        isReLU = true;
        RELU_range_in_place<m, k, Share, Datatype>(output, len);
        trunc_exact_opt_in_place<m, k, Share, Datatype>(output, len, true);
        isReLU = false;
#endif
    }
#if TS1_FUSED_ACTIVE && TS1_FOLD_POOL && FUSE_RELU_AVG == 1
    else if (curr_denom > 1)  // a pooling fused into a ReLU on a truncated value: TS1 divides
        RELU_range_in_place_ts1<m, k, Share, Datatype>(output, len);
#endif
    else
        RELU_range_in_place<m, k, Share, Datatype>(output, len);
#else
        RELU_range_in_place<m, k, Share, Datatype>(output, len);
#endif

#if TRUNC_DELAYED == 1
    delayed = false;
#endif

#if TRUNC_APPROACH > 0
    all_positive = true;
#endif
}
