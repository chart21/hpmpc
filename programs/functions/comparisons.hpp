#pragma once
#include "share_conversion.hpp"



template<int m, int k,typename Share, typename Datatype>
void LTZ(sint_t<Additive_Share<Datatype, Share>>* val, sint_t<Additive_Share<Datatype, Share>>* result, const int len)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using Bitset = sbitset_t<k-m, S>;
    using sint = sint_t<A>;
    S* y = new S[len]; 
    get_msb_range<0, k-m, Datatype, Share>(val, y, len);

    
    for(int i = 0; i < len; i++)
    {
        y[i].prepare_bit2a(result[i].get_share_pointer());
    }
    delete[] y;
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        result[i].complete_bit2a();
    }
    
}

template<int m, int k,typename Share, typename Datatype>
void EQZ(sint_t<Additive_Share<Datatype, Share>>* val, sint_t<Additive_Share<Datatype, Share>>* result, const int len)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<k-m, S>;
    using sint = sint_t<A>;
   
    
    Share::communicate();
    S *y = new S[len];
    S *y_check = new S[len];
    auto val_check = new sint[len];
    for(int i = 0; i < len; i++)
    {
        val_check[i] = val[i] - sint(1);
    }
    A2B_range<m, k, Share, DATATYPE>(val, y, len);
    A2B_range<m, k, Share, DATATYPE>(val_check, y_check, len);
    for(int i = 0; i < len; i++)
    {
        y[i] = y[i] ^ y_check[i];
    }

    delete[] y_check;

    for(int i = 0; i < len; i++)
    {
        y[i].prepare_bit2a(result[i].get_share_pointer());
    }
    delete[] y;
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        result[i].complete_bit2a();
    }


}

