#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include "../../protocols/Protocols.h"
#include "adders/rca.hpp"
#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
#include "adders/rca_msb.hpp"
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1
#include "adders/ppa_msb_4_way.hpp"
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0
#include "adders/ppa_msb_unsafe.hpp"
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

#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
    std::vector<BooleanAdder_MSB<bk - bm, S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0
    /* std::vector<PPA_MSB<bk-bm,S>> adders; */
    std::vector<PPA_MSB_Unsafe<bk - bm, S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1
    std::vector<PPA_MSB_4Way<bk - bm, S>> adders;
#endif

    adders.reserve(len);
    for (int i = 0; i < len; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], msb[i]);
    }
    while (!adders[0].is_done())
    {
        for (int i = 0; i < len; i++)
        {
            adders[i].step();
        }
        Share::communicate();
    }
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

    std::vector<BooleanAdder<bk - bm, S>> adders;

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
    std::vector<BooleanAdder<bk - bm, S>> adders2;

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
        y[i].prepare_opt_bit_injection(val[i].get_share_pointer(), val[i].get_share_pointer());
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
        tmp[counter++] = sint::load_shares(input + counter * BITLENGTH);
        m -= BITLENGTH;
    }
    if (m > 0)
        tmp[counter++] = sint::load_shares(m, input + counter * BITLENGTH);
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
        tmp[counter++] = sint::load_shares(input + counter * BITLENGTH);
        m -= BITLENGTH;
    }
    if (m > 0)
        tmp[counter++] = sint::load_shares(m, input + counter * BITLENGTH);
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
        tmp[counter++] = sint::load_shares(val + counter * BITLENGTH);
        m -= BITLENGTH;
    }
    if (m > 0)
        tmp[counter++] = sint::load_shares(m, val + counter * BITLENGTH);
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
        tmp[counter++] = sint::load_shares(val + counter * BITLENGTH);
        m -= BITLENGTH;
    }
    if (m > 0)
        tmp[counter++] = sint::load_shares(m, val + counter * BITLENGTH);
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
