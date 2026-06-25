#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include "../../protocols/Protocols.h"
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
#if A_KNOWN_TO_EVALUATORS_OPT == 1
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
#if FUSE_RELU_AVG == 1 && (TRUNC_APPROACH == 4 || TRUNC_APPROACH == 0) && TRUNC_DELAYED == 0
#include "../../datatypes/float_fixed_converter.hpp"
#endif

// compute msbs of a range of arithemtic shares
template <int bm, int bk, typename Datatype, typename Share>
void get_msb_range(sint_t<Additive_Share<Datatype, Share>>* val, XOR_Share<Datatype, Share>* msb, int len)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using Bitset = sbitset_t<bk - bm, S>;
    using sint = sint_t<A>;
    Bitset* s1 = new Bitset[len];
    Bitset* s2 = new Bitset[len];
#if A2B_ROUND_OPT_SIM == 0 
    //Skip if we are simulating A2B with round optimization
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
#endif

#if DEBUG_A2B == 1
    // Reveal the boolean A2B operands S1 (=bits(m)) and S2 (=bits(-lambda)) and check S1+S2 == v.
    // Reveal is mirrored in both phases (prepare_reveal stores an output share in PRE) so rounds stay matched.
    {
        const int NLANES = DATTYPE / BITLENGTH;  // arithmetic SIMD lanes
        for (int i = 0; i < len; i++)
        {
            s1[i].prepare_reveal_to_all();
            s2[i].prepare_reveal_to_all();
            val[i].get_share_pointer()[0].prepare_reveal_to_all();
        }
        Share::communicate();
        if (current_phase == PHASE_LIVE && PARTY == 0)
            printf("[A2B CALL] len=%d  NLANES(arith)=%d  DATTYPE=%d BITLENGTH=%d\n", len, NLANES, DATTYPE, BITLENGTH);
        for (int i = 0; i < len; i++)
        {
            alignas(sizeof(DATATYPE)) UINT_TYPE r1[DATTYPE];
            alignas(sizeof(DATATYPE)) UINT_TYPE r2[DATTYPE];
            alignas(sizeof(DATATYPE)) UINT_TYPE vr[DATTYPE];
            s1[i].complete_reveal_to_all(r1);
            s2[i].complete_reveal_to_all(r2);
            val[i].get_share_pointer()[0].complete_reveal_to_all(vr);
            if (current_phase == PHASE_LIVE && i == 0)
            {
                DATATYPE myl = val[i].get_share_pointer()[0].get_mask();
                printf("[A2B MASK P%d] elem0 lane0 mask l=%u (0x%08x)\n",
                       PARTY, (unsigned)(((UINT_TYPE*)&myl)[0]), (unsigned)(((UINT_TYPE*)&myl)[0]));
            }
            if (current_phase == PHASE_LIVE && PARTY == 0 && i < 4)
            {
                for (int ln = 0; ln < NLANES; ln++)
                {
                    UINT_TYPE mm = r1[ln];
                    UINT_TYPE nl = r2[ln];
                    UINT_TYPE exp_s2 = (UINT_TYPE)(vr[ln] - mm);
                    printf("  [i=%d lane=%d] v=%lld m=%lld S2=%lld expS2=%lld S2ok=%d\n",
                           i, ln, (long long)(INT_TYPE)vr[ln], (long long)(INT_TYPE)mm,
                           (long long)(INT_TYPE)nl, (long long)(INT_TYPE)exp_s2, (int)(nl == exp_s2));
                }
            }
        }
    }
#endif


std::vector<ADDER_TYPE<bk - bm, S>> adders;
    adders.reserve(len);
    for (int i = 0; i < len; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], msb[i]);
    }
   
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
    while (!adders[0].is_done())
    {
        for (int i = 0; i < len; i++)
        {
            adders[i].step();
        }
        Share::communicate();
    }
#endif
#if DEBUG_A2B == 1
    // Reliable check: reveal the ADDER's MSB output and compare to sign(v). msb's PRE share == online share
    // (the adder is deterministic), so this reveal is trustworthy (unlike the S2 reveal above).
    {
        for (int i = 0; i < len; i++) {
            msb[i].prepare_reveal_to_all();
            val[i].get_share_pointer()[0].prepare_reveal_to_all();
        }
        Share::communicate();
        int wrong = 0;
        for (int i = 0; i < len; i++) {
            DATATYPE mb = msb[i].complete_reveal_to_all();
            alignas(sizeof(DATATYPE)) UINT_TYPE vr[DATTYPE];
            val[i].get_share_pointer()[0].complete_reveal_to_all(vr);
            if (current_phase == PHASE_LIVE && PARTY == 0) {
                int sign_bit = (int)(((UINT_TYPE*)&mb)[0] & 1);
                int expected = ((INT_TYPE)vr[0] < 0) ? 1 : 0;
                if (sign_bit != expected) wrong++;
                if (i < 6) printf("[MSB i=%d] v=%lld msb=%d expected=%d %s\n",
                    i, (long long)(INT_TYPE)vr[0], sign_bit, expected, sign_bit == expected ? "OK" : "WRONG");
            }
        }
        if (current_phase == PHASE_LIVE && PARTY == 0) printf("[MSB] wrong=%d/%d\n", wrong, len);
    }
#endif
    delete[] s1;
    delete[] s2;
    adders.clear();
    adders.shrink_to_fit();
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

template <typename Datatype, typename Share>
void bit_injection_opt_range(XOR_Share<Datatype, Share>* y, sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    for (int i = 0; i < len; i++)
    {
#if FUSE_RELU_AVG == 1 && (TRUNC_APPROACH == 4 || TRUNC_APPROACH == 0) && TRUNC_DELAYED == 0
        auto reciprocal = FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1 / FLOATTYPE(curr_denom));
        y[i].prepare_opt_bit_injection_with_trunc(val[i].get_share_pointer(), val[i].get_share_pointer(), PROMOTE(reciprocal));
#else
        y[i].prepare_opt_bit_injection(val[i].get_share_pointer(), val[i].get_share_pointer());
#endif
    }
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
        val[i].complete_opt_bit_injection();
    }
}

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
