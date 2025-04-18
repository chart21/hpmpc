#pragma once
#include "share_conversion.hpp"
#if TRUNC_APPROACH > 0
#include "exact_truncation.hpp"
#endif
#include "prob_truncation.hpp"
/* #include "boolean_adder_bandwidth.hpp" */

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

    S* y = new S[len];
    get_msb_range<m, k, Datatype, Share>(val, y, len);

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
#if TRUNC_APPROACH == 0
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

template <int m, int k, typename Share, typename Datatype>
void RELU_range_in_place_optB2A(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<k - m, S>;
    using sint = sint_t<A>;

    S* y = new S[len];
    get_msb_range<m, k, Datatype, Share>(val, y, len);
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

template <int rm = 0, int rk = BITLENGTH, typename Share, typename Datatype>
static void RELU(const Additive_Share<Datatype, Share>* begin,
                 const Additive_Share<Datatype, Share>* end,
                 Additive_Share<Datatype, Share>* output)
{
    const int len = end - begin;
#if TRUNC_DELAYED == 1 && TRUNC_APPROACH > 0
    if (delayed)
    {
#if TRUNC_APPROACH == 2
        pack_additive_inplace<rm, rk>(begin, output, len, RELU_range_in_place_exact<rm, rk, Share, Datatype>);
#else
        isReLU = true;
        std::copy(begin, end, output);
        trunc_exact_opt_in_place(output, len, true);
        isReLU = false;
#endif
    }
    else
        pack_additive_inplace<rm, rk>(begin, output, len, RELU_range_in_place_opt<rm, rk, Share, Datatype>);
#else
    pack_additive_inplace<rm, rk>(begin, output, len, RELU_range_in_place_opt<rm, rk, Share, Datatype>);
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
#else
        isReLU = true;
        RELU_range_in_place_opt<m, k, Share, Datatype>(output, len);
        trunc_exact_opt_in_place<m, k, Share, Datatype>(output, len, true);
        isReLU = false;
#endif
    }
    else
        RELU_range_in_place_opt<m, k, Share, Datatype>(output, len);
#else
    RELU_range_in_place_opt<m, k, Share, Datatype>(output, len);
#endif

#if TRUNC_DELAYED == 1
    delayed = false;
#endif

#if TRUNC_APPROACH > 0
    all_positive = true;
#endif
}
