#pragma once
#include "share_conversion.hpp"
#include "../functions/adders/rca_msb_carry.hpp"
/* static void trunc_exact_in_place(sint_t<Additive_Share<Datatype, Share>>* val, const int len) */
/*     template<int m, int k,typename Share, typename Datatype> */
template<typename Datatype, typename Share, typename dummy>
void trunc_exact_in_place(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    const int bm = 0;
    const int bk = BITLENGTH;
    using Bitset = sbitset_t<bk-bm, S>;
    using sint = sint_t<A>;
   
    Bitset *y = new Bitset[len];
    A2B_range<bm,bk,Datatype,Share>(val, y, len);
    
    for(int i = 0; i < len; i++)
    {
    for(int j = BITLENGTH - 1; j >= FRACTIONAL; j--)
    {
        y[i][j] = y[i][j - FRACTIONAL]; //shift right
    }
    for(int j = 1; j < FRACTIONAL; j++)
    {
       y[i][j] = y[i][0]; //set most significant bits to sign bit
    }
    }
    B2A_range<bm,bk,Datatype,Share>(y, val, len);  
}

template<typename Datatype, typename Share>
void trunc_exact_in_place(Additive_Share<Datatype, Share>* val, const int len)
{
    pack_additive_inplace<0,BITLENGTH>(val,len,trunc_exact_in_place<Datatype,Share,void>);
}



template<typename Datatype, typename Share, typename dummy>
void trunc_exact_opt_in_place(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    const int bm = 0;
    const int bk = BITLENGTH;
    using Bitset = sbitset_t<bk-bm, S>;
    using Bitset_t = sbitset_t<FRACTIONAL-bm, S>;
    using sint = sint_t<A>;

    //step1: create x/2t
    sint* x2t = new sint[len];
    for(int i = 0; i < len; i++)
    {
            x2t[i] = val[i].prepare_mult_public_fixed(UINT_TYPE(1));
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        x2t[i].complete_public_mult_fixed();
    }
    // step2: create x mod 2t ^ B
    sint* xmod2t = new sint[len];
    for(int i = 0; i < len; i++)
    {
        xmod2t[i] = val[i].prepare_trunc_exact_xmod2t();
    }
    //step 3: Nothing to do
    Bitset *x_s1 = new Bitset[len];
    Bitset *x_s2 = new Bitset[len];
    Bitset_t *xmod2t_s1 = new Bitset_t[len];
    Bitset_t *xmod2t_s2 = new Bitset_t[len];
    for(int i = 0; i < len; i++)
    {
        x_s1[i] = Bitset::prepare_A2B_S1(bm, (S*) val[i].get_share_pointer());
        x_s2[i] = Bitset::prepare_A2B_S2(bm, (S*) val[i].get_share_pointer());
        xmod2t_s1[i] = Bitset_t::prepare_A2B_S1(bm, (S*) xmod2t[i].get_share_pointer());
        xmod2t_s2[i] = Bitset_t::prepare_A2B_S2(bm, (S*) xmod2t[i].get_share_pointer());
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        x_s1[i].complete_A2B_S1();
        x_s2[i].complete_A2B_S2();
        xmod2t_s1[i].complete_A2B_S1();
        xmod2t_s2[i].complete_A2B_S2();
    }

    // Step 4: Calculate carry bit of B1
    
    // Step 5: Caclulate carry bit of B2
    std::vector<BooleanAdder_MSB_Carry<FRACTIONAL-bm,S>> b1adder;
    std::vector<BooleanAdder_MSB_Carry<bk-bm,S>> b2adder;
#if TRUNC_APPROACH == 2 && TRUNC_DELAYED == 1 
        S* b2y;
    if (isReLU)
        b2y = new S[len];
#endif
    S* b1c = new S[len];
    S* b2c = new S[len]; 
   
    b1adder.reserve(len);
    b2adder.reserve(len);
    for(int i = 0; i < len; i++)
    {
        b1adder.emplace_back(xmod2t_s1[i], xmod2t_s2[i]);
        b2adder.emplace_back(x_s1[i], x_s2[i]);
    }

    while(!b1adder[0].is_done())
    {
        for(int i = 0; i < len; i++)
        {
            b1adder[i].step();
            b2adder[i].step();
        }
        /* std::cout << "Adder step ..." << std::endl; */
        Share::communicate();
    }
    delete[] xmod2t;
    delete[] xmod2t_s1;
    delete[] xmod2t_s2;
    for(int i = 0; i < len; i++)
    {
        b1c[i] = b1adder[i].get_carry();
    }
    b1adder.clear();
    b1adder.shrink_to_fit();
    while(!b2adder[0].is_done())
    {
        for(int i = 0; i < len; i++)
        {
            b2adder[i].step();
        }
        Share::communicate();
    }
    delete[] x_s1;
    delete[] x_s2;
    for(int i = 0; i < len; i++)
    {
        b2c[i] = b2adder[i].get_carry();
#if TRUNC_APPROACH == 2 && TRUNC_DELAYED == 1 
    if (isReLU)
        b2y[i] = ~ b2adder[i].get_msb();
#endif
    }
    b2adder.clear();
    b2adder.shrink_to_fit();

    /* // Step 6: Bit2A */
    sint* c1A = new sint[len];
    sint* c2A = new sint[len];
for (int i = 0; i < len; i++)
{
    b1c[i].prepare_bit2a(c1A[i].get_share_pointer());
    b2c[i].prepare_bit2a(c2A[i].get_share_pointer());
}
Share::communicate();
for (int i = 0; i < len; i++)
{
    c1A[i].complete_bit2a();
    c2A[i].complete_bit2a();
}

// Step 7: Output x/2t + b1A + b2 * 2^l-t
for (int i = 0; i < len; i++)
{
    val[i] = x2t[i] + c1A[i] + c2A[i].mult_public(UINT_TYPE(1) << (BITLENGTH - FRACTIONAL)); //in case of ReLU, bit inj
}
#if TRUNC_APPROACH == 2 && TRUNC_DELAYED == 1 // Compute ReLU
if (isReLU)
{
    bit_injection_opt_range(b2y, val, len);
    delete[] b2y;
}
#endif

delete[] x2t;
delete[] b1c;
delete[] b2c;
delete[] c1A;
delete[] c2A;


}

template<typename Datatype, typename Share>
void trunc_exact_opt_in_place(Additive_Share<Datatype, Share>* val, const int len)
{
    pack_additive_inplace<0,BITLENGTH>(val,len,trunc_exact_opt_in_place<Datatype,Share,void>);
}

