#pragma once
#include "../functions/adders/rca_msb_carry.hpp"
#include "share_conversion.hpp"
/* static void trunc_exact_in_place(sint_t<Additive_Share<Datatype, Share>>* val, const int len) */
/*     template<int m, int k,typename Share, typename Datatype> */
template <typename Datatype, typename Share, typename dummy>
void trunc_exact_in_place(sint_t<Additive_Share<Datatype, Share>>* val, const int len, int fractional_bits = FRACTIONAL)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    const int bm = 0;
    const int bk = BITLENGTH;
    using Bitset = sbitset_t<bk - bm, S>;
    using sint = sint_t<A>;

    Bitset* y = new Bitset[len];
    A2B_range<bm, bk, Datatype, Share>(val, y, len);

    for (int i = 0; i < len; i++)
    {
        for (int j = BITLENGTH - 1; j >= fractional_bits; j--)
        {
            y[i][j] = y[i][j - fractional_bits];  // shift right
        }
        for (int j = 1; j < fractional_bits; j++)
        {
            y[i][j] = y[i][0];  // set most significant bits to sign bit
        }
    }
    B2A_range<bm, bk, Datatype, Share>(y, val, len);
}

template <typename Datatype, typename Share>
void trunc_exact_in_place(Additive_Share<Datatype, Share>* val, const int len, int fractional_bits = FRACTIONAL)
{
    pack_additive_inplace<0, BITLENGTH>(val, len, fractional_bits, trunc_exact_in_place<Datatype, Share, void>);
}

template <typename Datatype, typename Share, typename dummy>
void trunc_exact_opt_in_place(sint_t<Additive_Share<Datatype, Share>>* val,
                              const int len,
                              int fractional_bits = FRACTIONAL)
{
    assert(fractional_bits == FRACTIONAL);
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    const int bm = 0;
    const int bk = BITLENGTH;
    const int bmm = BITLENGTH - FRACTIONAL;
    const int bkk = BITLENGTH;
    using Bitset = sbitset_t<bk - bm, S>;
    using Bitset_t = sbitset_t<bkk - bmm, S>;
    using sint = sint_t<A>;

    // step1: create x/2t
    sint* x2t = new sint[len];
    for (int i = 0; i < len; i++)
    {
        x2t[i] = val[i].prepare_trunc_share(fractional_bits);
    }
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
        x2t[i].complete_public_mult_fixed();
    }
    // step2: create x mod 2t ^ B
#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3
    sint* xmod2t = new sint[len];
    for (int i = 0; i < len; i++)
    {
        xmod2t[i] = val[i].prepare_trunc_exact_xmod2t(fractional_bits);
    }
#endif
    // step 3: Nothing to do
    Bitset* x_s1 = new Bitset[len];
    Bitset* x_s2 = new Bitset[len];
#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3  // one-off error only needs to be corrected for deterministic truncation
    Bitset_t* xmod2t_s1 = new Bitset_t[len];
    Bitset_t* xmod2t_s2 = new Bitset_t[len];
#endif
    for (int i = 0; i < len; i++)
    {
        x_s1[i] = Bitset::prepare_A2B_S1(bm, (S*)val[i].get_share_pointer());
        x_s2[i] = Bitset::prepare_A2B_S2(bm, (S*)val[i].get_share_pointer());
#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3
        xmod2t_s1[i] = Bitset_t::prepare_A2B_S1(bmm, (S*)xmod2t[i].get_share_pointer());
        xmod2t_s2[i] = Bitset_t::prepare_A2B_S2(bmm, (S*)xmod2t[i].get_share_pointer());
#endif
    }
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
        x_s1[i].complete_A2B_S1();
        x_s2[i].complete_A2B_S2();
#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3
        xmod2t_s1[i].complete_A2B_S1();
        xmod2t_s2[i].complete_A2B_S2();
#endif
    }

    // Step 4: Calculate carry bit of B1

    // Step 5: Caclulate carry bit of B2
#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3
    std::vector<BooleanAdder_MSB_Carry<bkk - bmm, S>> b1adder;
    b1adder.reserve(len);
    S* b1c = new S[len];
#endif

    std::vector<BooleanAdder_MSB_Carry<bk - bm, S>> b2adder;
    b2adder.reserve(len);
    S* b2c = new S[len];
#if TRUNC_DELAYED == 1
    S* b2y;
    if (isReLU)
        b2y = new S[len];
#endif

    for (int i = 0; i < len; i++)
    {
#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3
        b1adder.emplace_back(xmod2t_s1[i], xmod2t_s2[i]);
#endif
        b2adder.emplace_back(x_s1[i], x_s2[i]);
    }
#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3
    while (!b1adder[0].is_done())
    {
        for (int i = 0; i < len; i++)
        {
            b1adder[i].step();
            b2adder[i].step();
        }
        Share::communicate();
    }
    delete[] xmod2t;
    delete[] xmod2t_s1;
    delete[] xmod2t_s2;
    for (int i = 0; i < len; i++)
    {
        b1c[i] = b1adder[i].get_carry();
    }
    b1adder.clear();
    b1adder.shrink_to_fit();
#endif
    while (!b2adder[0].is_done())
    {
        for (int i = 0; i < len; i++)
        {
            b2adder[i].step();
        }
        Share::communicate();
    }
    for (int i = 0; i < len; i++)
    {
        b2c[i] = b2adder[i].get_carry();
#if TRUNC_DELAYED == 1
        if (isReLU)
            b2y[i] = ~b2adder[i].get_msb();
#endif
    }
    b2adder.clear();
    b2adder.shrink_to_fit();

    delete[] x_s1;
    delete[] x_s2;

    /* // Step 6: Bit2A */
#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3
    sint* c1A = new sint[len];
#endif
    sint* c2A = new sint[len];
    for (int i = 0; i < len; i++)
    {
#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3
        b1c[i].prepare_bit2a(c1A[i].get_share_pointer());
#endif
        b2c[i].prepare_bit2a(c2A[i].get_share_pointer());
    }
    Share::communicate();
    for (int i = 0; i < len; i++)
    {
#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3
        c1A[i].complete_bit2a();
#endif
        c2A[i].complete_bit2a();
    }

    /* #if TRUNC_DELAYED == 1 // Compute ReLU */
    /* if(isReLU) */
    /* { */
    /*     get_msb_range<bm,bk,Datatype,Share>(val, b2y, len); */
    /*     for (int i = 0; i < len; i++) */
    /*     { */
    /*         b2y[i] = ~ b2y[i]; */
    /*     } */
    /* } */
    /* #endif */

    // Step 7: Output x/2t + b1A - b2 * 2^l-t
    for (int i = 0; i < len; i++)
    {
#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3
        val[i] = x2t[i] + c1A[i] - c2A[i].mult_public(UINT_TYPE(1) << (BITLENGTH - fractional_bits));
#else
        val[i] = x2t[i] - c2A[i].mult_public(UINT_TYPE(1) << (BITLENGTH - fractional_bits));
#endif
    }
#if TRUNC_DELAYED == 1  // Compute ReLU
    if (isReLU)
    {
        bit_injection_opt_range(b2y, val, len);
        delete[] b2y;
    }
#endif

#if TRUNC_APPROACH == 2 || TRUNC_APPROACH == 3
    delete[] b1c;
    delete[] c1A;
#endif
    delete[] x2t;
    delete[] b2c;
    delete[] c2A;
}

template <typename Datatype, typename Share>
void trunc_exact_opt_in_place(Additive_Share<Datatype, Share>* val,
                              const int len,
                              bool isPositive = false,
                              int fractional_bits = FRACTIONAL)
{
    using A = Additive_Share<Datatype, Share>;
#if MSB0_OPT == 1  // if optimization is deactivaated, always add 2^l-1 to gurantee positive number
    if (!isPositive)
#endif
        for (int i = 0; i < len; i++)
            val[i] = val[i] + A((UINT_TYPE(1) << (BITLENGTH - 1)));  // add 2^l-1 to gurantee positive number

    pack_additive_inplace<0, BITLENGTH>(val, len, fractional_bits, trunc_exact_opt_in_place<Datatype, Share, void>);

#if MSB0_OPT == 1
    if (!isPositive)
#endif
        for (int i = 0; i < len; i++)
            val[i] = val[i] - A((UINT_TYPE(1) << (BITLENGTH - fractional_bits -
                                                  1)));  // substract 2^l-1 to reverse previous addition ..
}

/* template<typename Datatype, typename Share, typename dummy> */
/* static void trunc_2k_in_place(sint_t<Additive_Share<Datatype, Share>>* val, const int len, int fractional_bits =
 * FRACTIONAL) */
/* { */
/*     using S = XOR_Share<Datatype, Share>; */
/*     using A = Additive_Share<Datatype, Share>; */
/*     const int bm = 0; */
/*     const int bk = BITLENGTH; */
/*     const int bmm = BITLENGTH - FRACTIONAL; */
/*     const int bkk = BITLENGTH; */
/*     using Bitset = sbitset_t<bk-bm, S>; */
/*     using Bitset_t = sbitset_t<bkk-bmm, S>; */
/*     using sint = sint_t<A>; */
/*     using T = sint_t<A>; */

/*     T* r_msb = new T[len]; */
/*     T* r_mk2 = new T[len]; */
/*     T* c = new T[len]; */
/*     T* c_prime = new T[len]; */
/*     T::communicate(); */
/*     sint* xmod2t = new sint[len]; */
/*     sint* xmod2t2 = new sint[len]; */
/*     for(int i = 0; i < len; i++) */
/*     { */
/*         xmod2t[i] = val[i].prepare_trunc_exact_xmod2t(fractional_bits); */
/* #if PARTY == 0 */
/*         xmod2t2[i] = sint(0) - val[i]; */
/*         xmod2t2[i] = xmod2t2[i].prepare_trunc_exact_xmod2t(fractional_bits); */
/* #else */
/*         xmod2t2[i] = val[i].prepare_trunc_exact_xmod2t(fractional_bits); */
/* #endif */
/*     } */
/*     for(int i = 0; i < len; i++) */
/*     { */
/*         val[i].prepare_trunc_2k_inputs(r_mk2[i], r_msb[i], c[i], c_prime[i], fractional_bits); */
/*     } */
/*     T::communicate(); */
/*     for(int i = 0; i < len; i++) */
/*     { */
/*         val[i].complete_trunc_2k_inputs(r_mk2[i], r_msb[i],c[i], c_prime[i]); */
/*     } */
/*     T::communicate(); */
/*     T* b = new T[len]; */
/*     for(int i = 0; i < len; i++) */
/*         b[i].prepare_XOR(r_msb[i],c[i]); */
/*     T::communicate(); */
/*     for(int i = 0; i < len; i++) */
/*     { */
/*         b[i].complete_XOR(r_msb[i],c[i]); */
/*         val[i] = c_prime[i] + b[i].mult_public(UINT_TYPE(1) << (BITLENGTH - fractional_bits - 1)) - r_mk2[i]; */

/*     } */
/*     T::communicate(); */
/*     delete[] c; */

/*     delete[] r_mk2; */
/*     delete[] c_prime; */

/*     Bitset_t *xmod2t_s1 = new Bitset_t[len]; */
/*     Bitset_t *xmod2t_s2 = new Bitset_t[len]; */
/*     for(int i = 0; i < len; i++) */
/*     { */
/*         xmod2t_s1[i] = Bitset_t::prepare_A2B_S1(bmm, (S*) xmod2t[i].get_share_pointer()); */
/*         xmod2t_s2[i] = Bitset_t::prepare_A2B_S2(bmm, (S*) xmod2t[i].get_share_pointer()); */
/*     } */
/*     Share::communicate(); */
/*     for(int i = 0; i < len; i++) */
/*     { */
/*         xmod2t_s1[i].complete_A2B_S1(); */
/*         xmod2t_s2[i].complete_A2B_S2(); */
/*     } */

/*     // Step 4: Calculate carry bit of B1 */

/*     // Step 5: Caclulate carry bit of B2 */
/*     std::vector<BooleanAdder_MSB_Carry<bkk-bmm,S>> b1adder; */
/*     b1adder.reserve(len); */
/*     S* b1c = new S[len]; */

/*     for(int i = 0; i < len; i++) */
/*         b1adder.emplace_back(xmod2t_s1[i], xmod2t_s2[i]); */
/*     while(!b1adder[0].is_done()) */
/*     { */
/*         for(int i = 0; i < len; i++) */
/*         { */
/*             b1adder[i].step(); */
/*         } */
/*         Share::communicate(); */
/*     } */
/*     for(int i = 0; i < len; i++) */
/*     { */
/*         b1c[i] = b1adder[i].get_carry(); */
/*     } */
/*     b1adder.clear(); */
/*     b1adder.shrink_to_fit(); */
/*     sint* c1A = new sint[len]; */
/* for (int i = 0; i < len; i++) */
/* { */
/*     b1c[i].prepare_bit2a(c1A[i].get_share_pointer()); */
/* } */
/* Share::communicate(); */
/* for (int i = 0; i < len; i++) */
/*     c1A[i].complete_bit2a(); */

/* sint* c2A = new sint[len]; */
/* for(int i = 0; i < len; i++) */
/*     c2A[i] = c1A[i].prepare_mult(sint(1) - r_msb[i]); */
/* T::communicate(); */
/* for(int i = 0; i < len; i++) */
/* { */
/*     c2A[i].complete_mult_without_trunc(); */
/* } */

/* for (int i = 0; i < len; i++) */
/*     val[i] = val[i] + c2A[i]; */

/* for(int i = 0; i < len; i++) */
/*     { */
/*         xmod2t_s1[i] = Bitset_t::prepare_A2B_S1(bmm, (S*) xmod2t2[i].get_share_pointer()); */
/*         xmod2t_s2[i] = Bitset_t::prepare_A2B_S2(bmm, (S*) xmod2t2[i].get_share_pointer()); */
/*     } */
/*     Share::communicate(); */
/*     for(int i = 0; i < len; i++) */
/*     { */
/*         xmod2t_s1[i].complete_A2B_S1(); */
/*         xmod2t_s2[i].complete_A2B_S2(); */
/*     } */

/*     // Step 4: Calculate carry bit of B1 */

/*     // Step 5: Caclulate carry bit of B2 */
/*     std::vector<BooleanAdder_MSB_Carry<bkk-bmm,S>> b2adder; */
/*     b2adder.reserve(len); */

/*     for(int i = 0; i < len; i++) */
/*         b2adder.emplace_back(xmod2t_s1[i], xmod2t_s2[i]); */
/*     while(!b2adder[0].is_done()) */
/*     { */
/*         for(int i = 0; i < len; i++) */
/*         { */
/*             b2adder[i].step(); */
/*         } */
/*         Share::communicate(); */
/*     } */
/*     delete[] xmod2t; */
/*     delete[] xmod2t_s1; */
/*     delete[] xmod2t_s2; */
/*     for(int i = 0; i < len; i++) */
/*     { */
/*         b1c[i] = b2adder[i].get_carry(); */
/*     } */
/*     b2adder.clear(); */
/*     b2adder.shrink_to_fit(); */
/* for (int i = 0; i < len; i++) */
/* { */
/*     b1c[i].prepare_bit2a(c1A[i].get_share_pointer()); */
/* } */
/* Share::communicate(); */
/* for (int i = 0; i < len; i++) */
/*     c1A[i].complete_bit2a(); */

/* for(int i = 0; i < len; i++) */
/*     c2A[i] = c1A[i].prepare_mult(r_msb[i]); */
/* T::communicate(); */
/* for(int i = 0; i < len; i++) */
/* { */
/*     c2A[i].complete_mult_without_trunc(); */
/* } */

/* for (int i = 0; i < len; i++) */
/*     val[i] = val[i] + c2A[i]; */

/* delete[] b1c; */
/* delete[] c1A; */
/* } */

/* template<typename Datatype, typename Share> */
/* void trunc_2k_in_place(Additive_Share<Datatype, Share>* val, const int len, bool isPositive=false, int
 * fractional_bits = FRACTIONAL) */
/* { */
/*     using A = Additive_Share<Datatype, Share>; */
/*     using sint = sint_t<A>; */
/*     if(!isPositive) */
/*         for(int i = 0; i < len; i++) */
/*             val[i] = val[i] +  A((UINT_TYPE(1) << (BITLENGTH - 1))); // add 2^l-1 to gurantee positive number */

/*     /1* pack_additive_inplace<0,BITLENGTH>(val,len,fractional_bits,trunc_2k_in_place<sint,void>); *1/ */
/*     pack_additive_inplace<0,BITLENGTH>(val,len,fractional_bits,trunc_2k_in_place<Datatype,Share,void>); */

/*     if(!isPositive) */
/*         for(int i = 0; i < len; i++) */
/*             val[i] = val[i] - A((UINT_TYPE(1) << (BITLENGTH - fractional_bits - 1))); // substract 2^l-1 to reverse
 * previous addition .. */
/* } */
