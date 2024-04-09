#pragma once
#include "../protocols/Protocols.h"
#include "../../protocols/XOR_Share.hpp"
#include "../../protocols/Additive_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include <chrono>

/* #include "boolean_adder_bandwidth.hpp" */
#if TRUNC_APPROACH == 2
#include "boolean_adder_bandwidth.hpp"
#elif BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
#include "boolean_adder_msb.hpp"
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1
#include "ppa_msb_4_way.hpp"
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0
/* #include "ppa_msb.hpp" */
#include "ppa_msb_unsafe.hpp"
#endif
/* #include "ppa_msb.hpp" */
/* #include "ppa.hpp" */
/* #include "ppa_msb_unsafe.hpp" */
#if TTP_PROTOCOL == 1 && SIMULATE_MPC_FUNCTIONS == 0
template<int m, int k,typename Share, typename Datatype>
void RELU_range_in_place(sint_t<Additive_Share<Datatype, Share>>* val, int len)
{
    for(int i = 0; i < len; i++)
        val[i] = val[i].relu();
}


template<int m, int k,typename Share, typename Datatype>
void RELU_range_in_place(Additive_Share<Datatype, Share>* val, int len)
{
    for(int i = 0; i < len; i++)
        val[i] = val[i].relu();
}


#else

#if TRUNC_APPROACH == 2

    template<int m, int k,typename Share, typename Datatype>
void RELU_range_in_place_exact(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<k-m, S>;
    using sint = sint_t<A>;
   
    /* if(current_phase == 1) */
    /* std::cout << "RELU ..." << std::endl; */
    
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
    /* if(current_phase == 1) */
    /* std::cout << "A2B completed ..." << std::endl; */
    
    Share::communicate();
    /* if(current_phase == 1) */
    /* std::cout << "Adder ..." << std::endl; */

    /* Bitset* y = new Bitset[NUM_INPUTS]; */
    Bitset *y = new Bitset[len];
    /* BooleanAdder<S> *adder = new BooleanAdder<S>[NUM_INPUTS]; */
    std::vector<BooleanAdder<k-m,S>> adders;
    /* std::vector<PPA_MSB_4Way<k,S>> adders; */
    adders.reserve(len);
    for(int i = 0; i < len; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], y[i]);
    }

    while(!adders[0].is_done())
    {
        for(int i = 0; i < len; i++)
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
    
    for(int i = 0; i < len; i++)
    {
        auto msb = ~ y[i][0];
        for(int j = 0; j < k-m; j++)
        {
            y[i][j] = y[i][j] & msb;
        }
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        for(int j = 0; j < k-m; j++)
            y[i][j].complete_and();
    #if TRUNC_DELAYED == 1
    for(int j = BITLENGTH - 1; j >= FRACTIONAL; j--)
    {
        y[i][j] = y[i][j - FRACTIONAL]; //shift right
    }
    for(int j = 0; j < FRACTIONAL; j++)
    {
       y[i][j] = SET_ALL_ZERO(); //set most significant bits to zero
    }
    #endif
    }
    /* if(current_phase == 1) */
    /*     std::cout << "Bit inj ..." << std::endl; */
  
    Bitset* random_mask = new Bitset[len];
    for(int i = 0; i < len; i++)
    {
        for(int j = 0; j < k-m; j++)
        {
            random_mask[i][j].get_random_B2A();
        }
    }
    
    Bitset *z = new Bitset[len];
    std::vector<BooleanAdder<k-m,S>> adders2;
    /* std::vector<PPA_MSB_4Way<k,S>> adders; */
    adders2.reserve(len);
    for(int i = 0; i < len; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders2.emplace_back(y[i], random_mask[i], z[i]);
    }

    while(!adders2[0].is_done())
    {
        for(int i = 0; i < len; i++)
        {
            adders2[i].step();
        }
        /* std::cout << "Adder step ..." << std::endl; */
        Share::communicate();
    }
    adders2.clear();
    adders2.shrink_to_fit();
    delete[] y;
    for(int i = 0; i < len; i++)
    {
        /* sint::prepare_B2A((S*) z[i].get_share_pointer(), (S*) random_mask[i].get_share_pointer(), (S*) val[i].get_share_pointer()); */
        sint::prepare_B2A( z[i].get_share_pointer(), random_mask[i].get_share_pointer(), val[i].get_share_pointer());
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        sint::complete_B2A(z[i].get_share_pointer(), val[i].get_share_pointer());
    }
    delete[] z;
    delete[] random_mask;
}
#endif
    template<int m, int k,typename Share, typename Datatype>
void RELU_range_in_place_opt(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<k-m, S>;
    using sint = sint_t<A>;
   
    /* if(current_phase == 1) */
    /* std::cout << "RELU ..." << std::endl; */
    
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
    /* if(current_phase == 1) */
    /* std::cout << "A2B completed ..." << std::endl; */
    
    Share::communicate();
    /* if(current_phase == 1) */
    /* std::cout << "Adder ..." << std::endl; */

    /* Bitset* y = new Bitset[NUM_INPUTS]; */
    S *y = new S[len];
    /* BooleanAdder<S> *adder = new BooleanAdder<S>[NUM_INPUTS]; */
#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
    std::vector<BooleanAdder_MSB<k-m,S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1
    std::vector<PPA_MSB_4Way<k-m,S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0
    /* std::vector<PPA_MSB<k-m,S>> adders; */
    std::vector<PPA_MSB_Unsafe<k-m,S>> adders;
#endif
    /* std::vector<PPA_MSB_4Way<k,S>> adders; */
    adders.reserve(len);
    for(int i = 0; i < len; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], y[i]);
    }

    while(!adders[0].is_done())
    {
        for(int i = 0; i < len; i++)
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
    
    for(int i = 0; i < len; i++)
    {
        y[i] = ~ y[i];
    }
    /* if(current_phase == 1) */
    /*     std::cout << "Bit inj ..." << std::endl; */
    
    for(int i = 0; i < len; i++)
    {
        y[i].prepare_opt_bit_injection(val[i].get_share_pointer(),val[i].get_share_pointer());
    }
    delete[] y;
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        val[i].complete_opt_bit_injection();
    }
    
    Share::communicate();
#if TRUNC_DELAYED == 1
#if TRUNC_APPROACH == 0
    for(int i = 0; i < len; i++)
    {
        val[i].prepare_mult_public_fixed(UINT_TYPE(1)); //multiply by 1 to trigeger truncation
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        val[i].complete_public_mult_fixed();
    }
    Share::communicate();
#else
    trunc_2k_in_place(val, len);
#endif
#endif
    /* } */


}

    template<int m, int k,typename Share, typename Datatype>
void RELU_range_in_place_optB2A(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<k-m, S>;
    using sint = sint_t<A>;
   
    /* if(current_phase == 1) */
    /* std::cout << "RELU ..." << std::endl; */
    
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
    /* if(current_phase == 1) */
    /* std::cout << "A2B completed ..." << std::endl; */
    
    Share::communicate();
    /* if(current_phase == 1) */
    /* std::cout << "Adder ..." << std::endl; */

    /* Bitset* y = new Bitset[NUM_INPUTS]; */
    S *y = new S[len];
    /* BooleanAdder<S> *adder = new BooleanAdder<S>[NUM_INPUTS]; */
#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
    std::vector<BooleanAdder_MSB<k-m,S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1
    std::vector<PPA_MSB_4Way<k-m,S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0
    /* std::vector<PPA_MSB<k-m,S>> adders; */
    std::vector<PPA_MSB_Unsafe<k-m,S>> adders;
#endif
    /* std::vector<PPA_MSB_4Way<k,S>> adders; */
    adders.reserve(len);
    for(int i = 0; i < len; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], y[i]);
    }

    while(!adders[0].is_done())
    {
        for(int i = 0; i < len; i++)
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
    
    for(int i = 0; i < len; i++)
    {
        y[i] = ~ y[i];
    }
    /* if(current_phase == 1) */
    /*     std::cout << "Bit inj ..." << std::endl; */
    
    sint* result = new sint[len];
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
    
    Share::communicate();

    for(int i = 0; i < len; i++)
    {
        val[i] = result[i].prepare_dot(val[i]);
#if TRUNC_APPROACH == 0 && TRUNC_DELAYED == 1
        val[i].mask_and_send_dot();
#else
        val[i].mask_and_send_dot_without_trunc();
#endif 
    }
    delete[] result;
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
#if TRUNC_APPROACH == 0 && TRUNC_DELAYED == 1
        val[i].complete_mult();
#else
        val[i].complete_mult_without_trunc();
#endif
        /* val[i] -= sint(1); // To counter the +1 in TRUNC */
    }


    Share::communicate();


#if TRUNC_APPROACH == 1 && TRUNC_DELAYED == 1
    trunc_2k_in_place(val, len);
#endif


}

template<int m, int k,typename Share, typename Datatype>
void RELU_range_in_place(sint_t<Additive_Share<Datatype, Share>>* val, const int len)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<k-m, S>;
    using sint = sint_t<A>;
   
    /* if(current_phase == 1) */
    /* std::cout << "RELU ..." << std::endl; */
    
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
    /* if(current_phase == 1) */
    /* std::cout << "A2B completed ..." << std::endl; */
    
    Share::communicate();
    /* if(current_phase == 1) */
    /* std::cout << "Adder ..." << std::endl; */

    /* Bitset* y = new Bitset[NUM_INPUTS]; */
    S *y = new S[len];
    /* BooleanAdder<S> *adder = new BooleanAdder<S>[NUM_INPUTS]; */
#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
    std::vector<BooleanAdder_MSB<k-m,S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1
    std::vector<PPA_MSB_4Way<k-m,S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0
    /* std::vector<PPA_MSB<k-m,S>> adders; */
    std::vector<PPA_MSB_Unsafe<k-m,S>> adders;
#endif
    /* std::vector<PPA_MSB_4Way<k,S>> adders; */
    adders.reserve(len);
    for(int i = 0; i < len; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], y[i]);
    }

    while(!adders[0].is_done())
    {
        for(int i = 0; i < len; i++)
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
    
    for(int i = 0; i < len; i++)
    {
        y[i] = ~ y[i];
    }
    /* if(current_phase == 1) */
    /*     std::cout << "Bit inj ..." << std::endl; */
    
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
    /* if(current_phase == 1) */
    /*     std::cout << "XOR ..." << std::endl; */

#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
    
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
    /* if(current_phase == 1) */
    /*     std::cout << "MULT ..." << std::endl; */
    

    for(int i = 0; i < len; i++)
    {
        val[i] = result[i].prepare_dot(val[i]);
#if TRUNC_APPROACH == 0 && TRUNC_DELAYED == 1
        val[i].mask_and_send_dot();
#else
        val[i].mask_and_send_dot_without_trunc();
#endif 
    }
    delete[] result;
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
#if TRUNC_APPROACH == 0 && TRUNC_DELAYED == 1
        val[i].complete_mult();
#else
        val[i].complete_mult_without_trunc();
#endif
        /* val[i] -= sint(1); // To counter the +1 in TRUNC */
    }
    Share::communicate();

#else // MULTI_INPUT AND gate approach, TODO: Make compatible with non-interactive probabilistic truncation
    for(int i = 0; i < len; i++)
        val[i] = val[i].prepare_dot( t1[i] + t2[i]) - (val[i] + val[i]).prepare_dot3(t1[i],t2[i]); // (a+b) v - 2abv

#if TRUNC_APPROACH == 0 && TRUNC_DELAYED == 1
    for(int i = 0; i < len; i++)
        val[i].mask_and_send_dot(); // important: do not mix with prepare_dot in same loop because of send recv order
#else
    for(int i = 0; i < len; i++)
        val[i].mask_and_send_dot_without_trunc();
#endif 
    
    delete[] t1;
    delete[] t2;
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
#if TRUNC_APPROACH == 0 && TRUNC_DELAYED == 1
        /* t1[i].complete_mult(); */
        /* t2[i].complete_mult(); */
        /* val[i] = t1[i] - t2[i]; */
        val[i].complete_mult();
#else
        val[i].complete_mult_without_trunc();
#endif
    }
    
#endif

    Share::communicate();


#if TRUNC_APPROACH == 1 && TRUNC_DELAYED == 1
    trunc_2k_in_place(val, len);
#endif
    /* } */


}
    
#endif

template<typename T>
static void trunc_2k_in_place(T*  val, const int len){
    
    T* r_msb = new T[len];
    T* r_mk2 = new T[len];
    T* c = new T[len];
    T* c_prime = new T[len];
    T::communicate();
    for(int i = 0; i < len; i++)
    {
        val[i].prepare_trunc_2k_inputs(r_mk2[i], r_msb[i], c[i], c_prime[i]);
    }
    T::communicate();
    for(int i = 0; i < len; i++)
    {
        val[i].complete_trunc_2k_inputs(r_mk2[i], r_msb[i],c[i], c_prime[i]);
    }
    T::communicate();
    T* b = new T[len];
    for(int i = 0; i < len; i++)
        b[i].prepare_XOR(r_msb[i],c[i]);
    T::communicate();
    for(int i = 0; i < len; i++)
    {
        b[i].complete_XOR(r_msb[i],c[i]);
        b[i] = b[i].mult_public(UINT_TYPE(1) << (BITLENGTH - FRACTIONAL - 1));
    }
    T::communicate();
    delete[] c;
    
    for(int i = 0; i < len; i++)
    {
        val[i] = c_prime[i] + b[i] - r_mk2[i];
        /* val[i] = val[i] + T(1); */
    }
    delete[] r_mk2;
    delete[] r_msb;
    delete[] c_prime;
    delete[] b;
}


/* template<typename T> */
/* static void RELU(const T*  begin, const T* end, T*  output){ */
/*     std::copy(begin, end, output); */
/*     int len = end - begin; */
/*     /1* for (const sint_t* iter = begin; iter != end; ++iter) { *1/ */
/*             /1* output[i++] = iter->relu(); *1/ */
/*     RELU_range_in_place<REDUCED_BITLENGTH_m,REDUCED_BITLENGTH_k>(output, len); */
/*     /1* } *1/ */
/* } */


template<int rm = 0, int rk = BITLENGTH, typename Share, typename Datatype>
static void RELU(const Additive_Share<Datatype, Share>*  begin, const Additive_Share<Datatype, Share>* end, Additive_Share<Datatype, Share>*  output){
    using sint = sint_t<Additive_Share<Datatype, Share>>;
    /* std::copy(begin, end, output); */
    /* int m = end - begin; */
    /* int sint_len = ((m-1)/BITLENGTH)+1; */
    /* sint* tmp = new sint[sint_len]; */
    /* for(int i = 0; i < sint_len-1; i++) */
    /* { */
    /*     tmp[i] = sint::load_shares(begin+i*BITLENGTH); */
    /* } */
    /* tmp[sint_len-1] = sint::load_shares( m-((sint_len-1)*BITLENGTH), begin+(sint_len-1)*BITLENGTH); //leftover shares */
    /* /1* for (const sint_t* iter = begin; iter != end; ++iter) { *1/ */
    /*         /1* output[i++] = iter->relu(); *1/ */
    /* RELU_range_in_place<REDUCED_BITLENGTH_m,REDUCED_BITLENGTH_k,Share>(tmp, sint_len); */
    /* for(int i = 0; i < sint_len-1; i++) */
    /* { */
    /*     for(int j = 0; j < BITLENGTH; j++) */
    /*     { */
    /*         output[i*BITLENGTH+j] = tmp[i].get_share_pointer()[j]; */
    /*     } */
    /* } */
    
    /* for(int i = 0; i < m-((sint_len-1)*BITLENGTH); i++) */
    /* { */
    /*     output[(sint_len-1)*BITLENGTH+i] = tmp[sint_len-1].get_share_pointer()[i]; */
    /* } */
    /* delete[] tmp; */
    /* } */
    int m = end - begin;
    sint* tmp = new sint[(m-1)/BITLENGTH+1];
    int counter = 0;
    while(m > BITLENGTH-1)
    {
       tmp[counter++] = sint::load_shares(begin+counter*BITLENGTH);
       m -= BITLENGTH;
    }
    if(m > 0)
        tmp[counter++] = sint::load_shares(m, begin+counter*BITLENGTH);
    /* RELU_range_in_place<rm,rk,Share>(tmp, counter); */
#if TRUNC_APPROACH == 2
    RELU_range_in_place_exact<rm,rk,Share>(tmp, counter);
#else
    RELU_range_in_place_opt<rm,rk,Share>(tmp, counter);
#endif
    /* RELU_range_in_place_optB2A<rm,rk,Share>(tmp, counter); */
    /* for(int i = 0; i < counter; i++) */
    /* { */
        /* std::cout << tmp[i].get_p1() << std::endl; */
    /* } */
    counter = 0;
    m = end - begin;
    while(m > BITLENGTH-1)
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



template<int m = 0, int k = BITLENGTH, typename Share, typename Datatype>
static void RELU(const sint_t<Additive_Share<Datatype, Share>>*  begin, const sint_t<Additive_Share<Datatype, Share>>* end, sint_t<Additive_Share<Datatype, Share>>*  output){
    std::copy(begin, end, output);
    int len = end - begin;
    /* for (const sint_t* iter = begin; iter != end; ++iter) { */
            /* output[i++] = iter->relu(); */
    RELU_range_in_place<m,k,Share>(output, len);
    /* } */
}


