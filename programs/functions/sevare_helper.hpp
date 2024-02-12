#pragma once
#include "../../protocols/Protocols.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <bitset>
#include "../../protocols/XOR_Share.hpp"
#include "../../protocols/Additive_Share.hpp"
#include "../../protocols/Matrix_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include "boolean_adder_bandwidth.hpp"

#include "boolean_adder_msb.hpp"
#include "ppa_msb.hpp"
#include "ppa.hpp"
#include "ppa_msb_unsafe.hpp"
#include "ppa_msb_4_way.hpp"

#include "../../utils/print.hpp"

#include <cmath>
#include <iterator>
#include <algorithm>
#include <assert.h>
#if FUNCTION_IDENTIFIER == 43 || FUNCTION_IDENTIFIER == 45 || FUNCTION_IDENTIFIER == 46
#define BANDWIDTH_OPTIMIZED 1
#define ONLINE_OPTIMIZED 0
#elif FUNCTION_IDENTIFIER == 53 || FUNCTION_IDENTIFIER == 55 || FUNCTION_IDENTIFIER == 57
#define BANDWIDTH_OPTIMIZED 0
#define ONLINE_OPTIMIZED 0
#elif FUNCTION_IDENTIFIER == 54 || FUNCTION_IDENTIFIER == 56 || FUNCTION_IDENTIFIER == 58
#define BANDWIDTH_OPTIMIZED 0
#define ONLINE_OPTIMIZED 1
#endif


    template<typename Share, typename Datatype>
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

// compute msbs of a range of arithemtic shares
template<int bm, int bk, typename Datatype, typename Share>
void get_msb_range(sint_t<Additive_Share<Datatype, Share>>* val, XOR_Share<Datatype, Share>* msb, int len)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using Bitset = sbitset_t<bk-bm,S>;
using sint = sint_t<A>;
Bitset *s1 = new Bitset[len];
Bitset *s2 = new Bitset[len];
    for(int i = 0; i < len; i++)
    {
        s1[i] = Bitset::prepare_A2B_S1( (S*) val[i].get_share_pointer());
        s2[i] = Bitset::prepare_A2B_S2( (S*) val[i].get_share_pointer());
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        s1[i].complete_A2B_S1();
        s2[i].complete_A2B_S2();
    }

#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
    std::vector<BooleanAdder_MSB<bk-bm,S>> adders;
#elif ONLINE_OPTIMIZED == 1 && BANDWIDTH_OPTIMIZED == 0
    std::vector<PPA_MSB_4Way<bk-bm,S>> adders;
#elif ONLINE_OPTIMIZED == 0 && BANDWIDTH_OPTIMIZED == 0
    std::vector<PPA_MSB<bk-bm,S>> adders;
#endif
    
    adders.reserve(len);
    for(int i = 0; i < len; i++)
    {
        adders.emplace_back(s1[i], s2[i], msb[i]);
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
    

}

template<int bm, int bk, typename Datatype, typename Share>
void max_min_msb_range(sint_t<Additive_Share<Datatype, Share>>* val, XOR_Share<Datatype, Share>* msb,const int og_len, int m, int batch_size, bool want_max)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using sint = sint_t<A>;

       int len = (m+1)/2;
       sint* max_val = new sint[batch_size*len];
       int offset = m % 2; // if m is odd, offset is 1
       int counter = 0;
       for(int b = 0; b < batch_size; b++)
       {
       counter = 0;
       for(int j = 1; j < m; j+=2)
       {
           if(want_max)
                max_val[counter+b*len] = val[j+b*og_len] - val[j-1 + b*og_len];
           else
                max_val[counter+b*len] = val[j-1 +b*og_len] - val[j + b*og_len];
            counter++;
       }
       }
    get_msb_range<bm,bk>(max_val, msb, len*batch_size);

    delete[] max_val;

    // get arithmetic version of msb to update values
    auto max_idx = new sint[batch_size*len];
    bitinj_range(msb, batch_size*len, max_idx);
    for(int b = 0; b < batch_size; b++)
    { 
    for(int i = 0; i < counter; i++)
    {
        max_idx[i+b*len] = max_idx[i+b*len] * (val[2*i+b*og_len] - val[2*i+1 + b*og_len]);
    }
    }
    Share::communicate();
    for(int b = 0; b < batch_size; b++)
    { 
    for(int i = 0; i < counter; i++)
    {
        max_idx[b*len + i].complete_mult_without_trunc();
        max_idx[b*len + i] = max_idx[b*len + i] + val[b*og_len + 2*i+1];
        val[b*og_len + i] = max_idx[b*len + i];

    }
    }
       if(offset == 1)
       {
           for(int b = 0; b < batch_size; b++)
           {
            val[b*og_len + counter] = val[b*og_len + m-1]; // last uneven element is always pairwise max
            msb[b*len + counter] = SET_ALL_ONE();
       }
       }
    delete[] max_idx;

}
    
    template<int bm,int bk, typename Datatype, typename Share>
void max_min(const sint_t<Additive_Share<Datatype, Share>>* begin, int len, sint_t<Additive_Share<Datatype, Share>>* output, int batch_size, bool want_max)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using sint = sint_t<A>;
int og_len = len;
int m = len;

sint* val = new sint[batch_size*og_len];
std::copy(begin, begin+batch_size*og_len, val);
if(len == 1)
{
   for(int b = 0; b < batch_size; b++)
       output[b] = val[0];
   return;
}

int log2m = std::ceil(std::log2(m)); 
for(int i = 0; i < log2m; i++)
{
    int counter = m/2; // 
    int offset = m % 2; // if m is odd, offset is 1
    int q = (m+1)/2;
    S* msb = new S[batch_size*q];
    max_min_msb_range<bm,bk>(val,msb,og_len,m,batch_size,want_max); //get msb and max of 0 -> counter

    delete[] msb;
    m = counter + offset;
}
for(int b = 0; b < batch_size; b++)
    {
       output[b] = val[b*og_len];
    }
delete[] val;
}
    

    template<typename Protocol>
sbitset_t<BITLENGTH, XOR_Share<DATATYPE, Protocol>> intersect(const sbitset_t<BITLENGTH, XOR_Share<DATATYPE, Protocol>>* a, const sbitset_t<BITLENGTH, XOR_Share<DATATYPE, Protocol>>* b, sbitset_t<BITLENGTH, XOR_Share<DATATYPE, Protocol>>* result, const int len_a, const int len_b)
{
    using S = XOR_Share<DATATYPE, Protocol>;
    using Bitset = sbitset_t<BITLENGTH,S>;



    auto tmp = new Bitset[len_a * len_b];

    assert(len_a <= len_b);


    for(int i = 0; i < len_a; i++)
        for(int j = 0; j < len_b; j++)
            tmp[i*len_b + j] = ~ (a[i] ^ b[j]); // vals[i] == element
  
  for (int k = BITLENGTH >> 1; k > 0; k = k >> 1) {
    for (int i = 0; i < k; i++) {
        int j = i * 2;
      for (int s = 0; s < len_a; s++) {
          for(int t = 0; t < len_b; t++)
              tmp[s*len_b + t][i] = tmp[s*len_b + t][j] & tmp[s*len_b + t][j +1];
    }
    }
  

    Protocol::communicate(); 

    for (int i = 0; i < k; i++) {
      for (int s = 0; s < len_a; s++) {
          for(int t = 0; t < len_b; t++)
              tmp[s*len_b + t][i].complete_and();
      }
      


    }

    Protocol::communicate();

  }

  auto intersect = new S[len_a];

    for(int i = 0; i < len_a; i++)
    {
        intersect[i] = SET_ALL_ZERO();
        for (int j = 1; j < len_b; i++) 
          intersect[i] = intersect[i] ^ tmp[i*len_b + j][0]; //intersect is 1 if element has been found
    }
        
    for(int i = 0; i < len_a; i++)
        for (int k = 0; k < BITLENGTH; k++)
            result[i][k] = a[i][k] & intersect[i]; // store element if it has been found, otherwise 0
    Protocol::communicate();
    for(int i = 0; i < len_a; i++)
        for (int k = 0; k < BITLENGTH; k++)
            result[i][k].complete_and();


}

    template<typename Protocol>
sbitset_t<BITLENGTH, XOR_Share<DATATYPE, Protocol>> intersect_single(const sbitset_t<BITLENGTH, XOR_Share<DATATYPE, Protocol>>* vals, const sbitset_t<BITLENGTH, XOR_Share<DATATYPE, Protocol>> element, int len)
{
    using S = XOR_Share<DATATYPE, Protocol>;
    using Bitset = sbitset_t<BITLENGTH,S>;

    Bitset* dataset = new Bitset[len];
    for(int i = 0; i < len; i++)
    {
        dataset[i] = ~ (vals[i] ^ element); // vals[i] == element
    }
  
  for (int k = BITLENGTH >> 1; k > 0; k = k >> 1) {
    for (int i = 0; i < k; i++) {
        int j = i * 2;
      for (int s = 0; s < len; s++) {
          dataset[s][i] = dataset[s][j] & dataset[s][j +1]; 
      }
    }

    Protocol::communicate(); 

    for (int i = 0; i < k; i++) {
      for (int s = 0; s < len; s++) {
          dataset[s][i].complete_and();
      }
      


    }

    Protocol::communicate();

  }
 

  for (int i = 1; i < len; i++) {
    dataset[0][0] = dataset[i][0] ^ dataset[0][0]; //dataset is 1 if element has been found
  }

  Bitset result = element;
  for (int k = 0; k < BITLENGTH; k++)
    result[k] = result[k] & dataset[0][0];
    
  Protocol::communicate();

  for (int k = 0; k < BITLENGTH; k++)
    result[k].complete_and();
  return result;
}
    
    template<int bm, int bk, typename Datatype, typename Share>
void max_min_sint(const sint_t<Additive_Share<Datatype, Share>>* begin, int len, sint_t<Additive_Share<Datatype, Share>>* output, int batch_size, bool want_max)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using sint = sint_t<A>;
max_min<bm,bk>(begin, len, output, batch_size, want_max);
} 
    template<int bm, int bk, typename Datatype, typename Share>
void max_min_sint(const Additive_Share<Datatype, Share>* begin, int len, Additive_Share<Datatype, Share>* output,int batch_size, bool want_max)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using sint = sint_t<A>;
sint *tmp_output = new sint[((batch_size-1)/BITLENGTH+1)];
sint *tmp_begin = new sint[len*((batch_size-1)/BITLENGTH+1)];
int k = 0;
int c = 0;
for(int i = 0; i < len; i++)
{
    for(int j = 0; j < batch_size; j++)
    {
        tmp_begin[c*len+i][k++] = begin[j*len+i]; //load all shares at position i into one sint
        if(k == BITLENGTH)
        {
            k = 0;
            c++;
        } 
    }
    k = 0;
    c = 0;
}

max_min_sint<bm,bk>(tmp_begin, len, tmp_output,(batch_size-1)/BITLENGTH+1, want_max); //TODO: Warning because of massive overhead
    for(int j = 0; j < batch_size; j++)
    {
        output[j] = tmp_output[c][k++]; //load result into output
        if(k == BITLENGTH)
        {
            k = 0;
            c++;
        } 
    }
delete[] tmp_output;
delete[] tmp_begin;
}

