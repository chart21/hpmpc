#pragma once
#include "../protocols/Protocols.h"
#include "../../protocols/XOR_Share.hpp"
#include "../../protocols/Additive_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include <chrono>

/* #include "boolean_adder_bandwidth.hpp" */

#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
#include "boolean_adder_msb.hpp"
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1
#include "ppa_msb_4_way.hpp"
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0
#include "ppa_msb_unsafe.hpp"
#endif
template<int m, int k,typename Share, typename Datatype>
void DRELU_range_in_place(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<k-m, S>;
    using sint = sint_t<A>;
   
    
    Share::communicate();
    Bitset *s1 = new Bitset[len];
    Bitset *s2 = new Bitset[len];
    for(int i = 0; i < len; i++)
    {
        s1[i] = Bitset::prepare_A2B_S1(m, (S*) val[i].get_share_pointer());
        s2[i] = Bitset::prepare_A2B_S2(m, (S*) val[i].get_share_pointer());
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        s1[i].complete_A2B_S1();
        s2[i].complete_A2B_S2();
    }
    
    Share::communicate();

    S *y = new S[len];
#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
    std::vector<BooleanAdder_MSB<k-m,S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1
    std::vector<PPA_MSB_4Way<k-m,S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0
    std::vector<PPA_MSB_Unsafe<k-m,S>> adders;
#endif
    adders.reserve(len);
    for(int i = 0; i < len; i++)
    {
        adders.emplace_back(s1[i], s2[i], y[i]);
    }

    while(!adders[0].is_done())
    {
        for(int i = 0; i < len; i++)
        {
            adders[i].step();
        }
        Share::communicate();
    }
    delete[] s1;
    delete[] s2;
    adders.clear();
    adders.shrink_to_fit();
    
    for(int i = 0; i < len; i++)
    {
        y[i] = ~ y[i];
    }
    
    sint* t1 = new sint[len];
    sint* t2 = new sint[len];
    for(int i = 0; i < len; i++)
    {
        y[i].prepare_bit_injection_S1(t1[i].get_share_pointer());
        y[i].prepare_bit_injection_S2(t2[i].get_share_pointer());
    }
    delete[] y;
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        t1[i].complete_bit_injection_S1();
        t2[i].complete_bit_injection_S2();
    }
    
    Share::communicate();
    
    sint* result = new sint[len];
    for(int i = 0; i < len; i++)
    {
        result[i].prepare_XOR(t1[i],t2[i]);
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        result[i].complete_XOR(t1[i],t2[i]);
    }
    delete[] t1;
    delete[] t2;

    Share::communicate();
    

    for(int i = 0; i < len; i++)
    {
        val[i] = result[i];
    }
} 



template<int rm = 0, int rk = BITLENGTH, typename Share, typename Datatype>
static void DRELU(const Additive_Share<Datatype, Share>*  begin, const Additive_Share<Datatype, Share>* end, Additive_Share<Datatype, Share>*  output){
    using sint = sint_t<Additive_Share<Datatype, Share>>;
    int m = end - begin;
    sint* tmp = new sint[(m-1)/BITLENGTH+1];
    int counter = 0;
    while(m > 31)
    {
       tmp[counter++] = sint::load_shares(begin+counter*BITLENGTH);
       m -= BITLENGTH;
    }
    if(m > 0)
        tmp[counter++] = sint::load_shares(m, begin+counter*BITLENGTH);
    DRELU_range_in_place<rm,rk,Share>(tmp, counter);
    /* for(int i = 0; i < counter; i++) */
    /* { */
        /* std::cout << tmp[i].get_p1() << std::endl; */
    /* } */
    counter = 0;
    m = end - begin;
    while(m > 31)
    {
        for(int j = 0; j < BITLENGTH; j++)
        {
            /* output[counter*BITLENGTH+j] = tmp[counter].get_share_pointer()[j]; */
            output[counter*BITLENGTH+j] = tmp[counter].get_share(j);
        }
        counter++;
        m -= BITLENGTH;
    }
    if(m > 0)
    {
        for(int j = 0; j < m; j++)
        {
            output[counter*BITLENGTH+j] = tmp[counter].get_share_pointer()[j];
        }
    }
    delete[] tmp;
}



template<typename Share, typename Datatype,int m = 0, int k = BITLENGTH>
static void DRELU(const sint_t<Additive_Share<Datatype, Share>>*  begin, const sint_t<Additive_Share<Datatype, Share>>* end, sint_t<Additive_Share<Datatype, Share>>*  output){
    std::copy(begin, end, output);
    int len = end - begin;
    DRELU_range_in_place<m,k,Share>(output, len);
}



