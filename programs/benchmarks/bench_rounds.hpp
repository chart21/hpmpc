#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"

#define RESULTTYPE DATATYPE

#if FUNCTION_IDENTIFIER == 8 || FUNCTION_IDENTIFIER == 9
#define FUNCTION MULT_Round_Test  // Regular / Fixed point multiplication
#elif FUNCTION_IDENTIFIER == 10
#define FUNCTION A2Bit_Setup_Round_Test
#elif FUNCTION_IDENTIFIER == 11
#define FUNCTION BIT2A_Setup_Round_Test
#elif FUNCTION_IDENTIFIER == 12
#define FUNCTION dot_prod_round_bench
#endif

#if FUNCTION_IDENTIFIER == 8 || FUNCTION_IDENTIFIER == 9
template <typename Share>
void MULT_Round_Test(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    const int sequential_rounds = NUM_INPUTS;  // specify number of rounds
    auto a = new A[DATTYPE];
    auto b = new A[DATTYPE];

    Share::communicate();  // dummy round
    for (int i = 0; i < sequential_rounds; i++)
    {
        for (int i = 0; i < DATTYPE; i++)
        {
#if FUNCTION_IDENTIFIER == 8  // regular multiplication
            a[i] = a[i].prepare_mult(b[i]);
#elif FUNCTION_IDENTIFIER == 9  // fixed point multiplication
            a[i] = a[i].prepare_dot(b[i]);
            a[i].mask_and_send_dot();
#endif
        }
        Share::communicate();
        for (int i = 0; i < DATTYPE; i++)
        {
#if FUNCTION_IDENTIFIER == 8  // regular multiplication
            a[i].complete_mult_without_trunc();
#elif FUNCTION_IDENTIFIER == 9  // fixed point multiplication
            a[i].complete_mult();
#endif
        }
        Share::communicate();
    }
    a[0].prepare_reveal_to_all();  // dummy reveal
    Share::communicate();
    *res = a[0].complete_reveal_to_all();
}

#elif FUNCTION_IDENTIFIER == 10

template <typename Share>
void A2Bit_Setup_Round_Test(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    const int sequential_rounds = NUM_INPUTS;  // specify number of rounds
    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    using sint = sint_t<A>;

    Share::communicate();  // dummy round
    const int len = 1;
    Bitset* s1 = new Bitset[len];
    Bitset* s2 = new Bitset[len];
    sint* val = new sint[len];
    for (int q = 0; q < sequential_rounds; q++)
    {
        for (int i = 0; i < len; i++)
        {
            s1[i] = Bitset::prepare_A2B_S1((S*)val[i].get_share_pointer());
            s2[i] = Bitset::prepare_A2B_S2((S*)val[i].get_share_pointer());
        }
        Share::communicate();
        for (int i = 0; i < len; i++)
        {
            s1[i].complete_A2B_S1();
            s2[i].complete_A2B_S2();
        }
        Share::communicate();
    }
    A dummy;
    dummy.prepare_reveal_to_all();  // dummy reveal
    Share::communicate();
    *res = dummy.complete_reveal_to_all();
    delete[] s1;
    delete[] s2;
    delete[] val;
}

#elif FUNCTION_IDENTIFIER == 11
template <typename Share>
void BIT2A_Setup_Round_Test(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    const int sequential_rounds = NUM_INPUTS;  // specify number of rounds
    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    using sint = sint_t<A>;
    S* y = new S;
    sint* t1 = new sint;
    sint* t2 = new sint;
    for (int j = 0; j < sequential_rounds; j++)
    {
        for (int i = 0; i < 1; i++)
        {
            y[i].prepare_bit_injection_S1(t1[i].get_share_pointer());
            y[i].prepare_bit_injection_S2(t2[i].get_share_pointer());
        }
        Share::communicate();
        for (int i = 0; i < 1; i++)
        {
            t1[i].complete_bit_injection_S1();
            t2[i].complete_bit_injection_S2();
        }
    }
    delete y;
    delete t1;
    delete t2;
    A dummy;
    dummy.prepare_reveal_to_all();
    Share::communicate();
    *res = dummy.complete_reveal_to_all();
}

#elif FUNCTION_IDENTIFIER == 12

template <typename Share>
void dot_prod_round_bench(DATATYPE* res)
{
    using M = Additive_Share<DATATYPE, Share>;
    Share::communicate();                      // dummy round
    const int sequential_rounds = NUM_INPUTS;  // low number of rounds due to high computational complexity
    const int dot_prod_size = 20000;
    auto a = new M[dot_prod_size];
    auto b = new M[dot_prod_size][dot_prod_size];
    auto c = new M[dot_prod_size];
    for (int rounds = 0; rounds < sequential_rounds; rounds++)
    {
        for (int i = 0; i < dot_prod_size; i++)
        {
            for (int j = 0; j < dot_prod_size; j++)
            {
                c[i] += a[i].prepare_dot(b[j][i]);
            }
            c[i].mask_and_send_dot_without_trunc();
        }
        Share::communicate();
        for (int i = 0; i < dot_prod_size; i++)
        {
            c[i].complete_mult_without_trunc();
        }
        Share::communicate();
    }
    c[dot_prod_size - 1].prepare_reveal_to_all();
    Share::communicate();
    *res = c[dot_prod_size - 1].complete_reveal_to_all();

    delete[] a;
    delete[] b;
    delete[] c;
}
#endif
