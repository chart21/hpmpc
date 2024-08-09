#pragma once
#include "share_conversion.hpp"
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

