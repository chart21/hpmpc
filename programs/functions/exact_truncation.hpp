#pragma once
#include "share_conversion.hpp"
template<typename Datatype, typename Share>
static void trunc_exact_in_place(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
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
    }
    B2A_range<bm,bk,Datatype,Share>(y, val, len);  
}

