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
/* #include "ppa_msb.hpp" */
#include "ppa_msb_unsafe.hpp"
#endif


template<int m, int k,typename Share, typename Datatype>
void LTZ(sint_t<Additive_Share<Datatype, Share>>* val, sint_t<Additive_Share<Datatype, Share>>* result, const int len)
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
    

    /* sint* t1 = new sint[len]; */
    /* sint* t2 = new sint[len]; */
    /* for(int i = 0; i < len; i++) */
    /* { */
    /*     y[i].prepare_bit_injection_S1(t1[i].get_share_pointer()); */
    /*     y[i].prepare_bit_injection_S2(t2[i].get_share_pointer()); */
    /* } */
    /* delete[] y; */
    /* Share::communicate(); */
    /* for(int i = 0; i < len; i++) */
    /* { */
    /*     t1[i].complete_bit_injection_S1(); */
    /*     t2[i].complete_bit_injection_S2(); */
    /* } */
    
    /* Share::communicate(); */
    
    /* for(int i = 0; i < len; i++) */
    /* { */
    /*     result[i].prepare_XOR(t1[i],t2[i]); */
    /* } */
    /* Share::communicate(); */
    /* for(int i = 0; i < len; i++) */
    /* { */
    /*     result[i].complete_XOR(t1[i],t2[i]); */
    /* } */
    
    /* delete[] t1; */
    /* delete[] t2; */
    
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
   
    /* if(current_phase == 1) */
    /* std::cout << "RELU ..." << std::endl; */
    
    Share::communicate();
    auto val_check = new sint[len];
    for(int i = 0; i < len; i++)
    {
        val_check[i] = val[i] - sint(1);
    }
    Bitset *s1 = new Bitset[len];
    Bitset *s2 = new Bitset[len];
    Bitset *s1_check = new Bitset[len];
    Bitset *s2_check = new Bitset[len];
    for(int i = 0; i < len; i++)
    {
        s1[i] = Bitset::prepare_A2B_S1(m, (S*) val[i].get_share_pointer());
        s2[i] = Bitset::prepare_A2B_S2(m, (S*) val[i].get_share_pointer());
        s1_check[i] = Bitset::prepare_A2B_S1(m, (S*) val_check[i].get_share_pointer());
        s2_check[i] = Bitset::prepare_A2B_S2(m, (S*) val_check[i].get_share_pointer());
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        s1[i].complete_A2B_S1();
        s2[i].complete_A2B_S2();
        s1_check[i].complete_A2B_S1();
        s2_check[i].complete_A2B_S2();
    }
    /* if(current_phase == 1) */
    /* std::cout << "A2B completed ..." << std::endl; */
    
    Share::communicate();
    /* if(current_phase == 1) */
    /* std::cout << "Adder ..." << std::endl; */

    /* Bitset* y = new Bitset[NUM_INPUTS]; */
    S *y = new S[len];
    S *y_check = new S[len];
    /* BooleanAdder<S> *adder = new BooleanAdder<S>[NUM_INPUTS]; */
#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
    std::vector<BooleanAdder_MSB<k-m,S>> adders;
    std::vector<BooleanAdder_MSB<k-m,S>> adders_check;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1
    std::vector<PPA_MSB_4Way<k-m,S>> adders;
    std::vector<PPA_MSB_4Way<k-m,S>> adders_check;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0
    /* std::vector<PPA_MSB<k-m,S>> adders; */
    std::vector<PPA_MSB_Unsafe<k-m,S>> adders;
    std::vector<PPA_MSB_Unsafe<k-m,S>> adders_check;
#endif
    /* std::vector<PPA_MSB_4Way<k,S>> adders; */
    adders.reserve(len);
    adders_check.reserve(len);
    for(int i = 0; i < len; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], y[i]);
        adders_check.emplace_back(s1_check[i], s2_check[i], y_check[i]);
    }

    while(!adders[0].is_done())
    {
        for(int i = 0; i < len; i++)
        {
            adders[i].step();
            adders_check[i].step();
        }
        /* std::cout << "Adder step ..." << std::endl; */
        Share::communicate();
    }
    delete[] s1;
    delete[] s2;
    delete[] s1_check;
    delete[] s2_check;
    adders.clear();
    adders.shrink_to_fit();
    adders_check.clear();
    adders_check.shrink_to_fit();
    
    for(int i = 0; i < len; i++)
    {
        y[i] = y[i] ^ y_check[i];
    }
    /* if(current_phase == 1) */
    /*     std::cout << "Bit inj ..." << std::endl; */
    
    /* for(int i = 0; i < len; i++) */
    /* { */
    /*     y[i].prepare_opt_bit_injection(result[i].get_share_pointer(),result[i].get_share_pointer()); */
    /* } */
    /* delete[] y; */
    /* Share::communicate(); */
    /* for(int i = 0; i < len; i++) */
    /* { */
    /*     result[i].complete_opt_bit_injection(); */
    /* } */
    
    /* sint* t1 = new sint[len]; */
    /* sint* t2 = new sint[len]; */
    /* for(int i = 0; i < len; i++) */
    /* { */
    /*     y[i].prepare_bit_injection_S1(t1[i].get_share_pointer()); */
    /*     y[i].prepare_bit_injection_S2(t2[i].get_share_pointer()); */
    /* } */
    /* delete[] y; */
    /* Share::communicate(); */
    /* for(int i = 0; i < len; i++) */
    /* { */
    /*     t1[i].complete_bit_injection_S1(); */
    /*     t2[i].complete_bit_injection_S2(); */
    /* } */
    
    /* Share::communicate(); */
    
    /* for(int i = 0; i < len; i++) */
    /* { */
    /*     result[i].prepare_XOR(t1[i],t2[i]); */
    /* } */
    /* Share::communicate(); */
    /* for(int i = 0; i < len; i++) */
    /* { */
    /*     result[i].complete_XOR(t1[i],t2[i]); */
    /* } */
    
    /* delete[] t1; */
    /* delete[] t2; */

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

template<typename T>
static void trunc_pr(T*  input, T* output, const int len){
    std::copy(input, input + len, output);    
    T* r_msb = new T[len];
    T* r_mk2 = new T[len];
    T* c = new T[len];
    T* c_prime = new T[len];
    T::communicate();
    for(int i = 0; i < len; i++)
    {
        output[i].prepare_trunc_2k_inputs(r_mk2[i], r_msb[i], c[i], c_prime[i]);
    }
    T::communicate();
    for(int i = 0; i < len; i++)
    {
        output[i].complete_trunc_2k_inputs(r_mk2[i], r_msb[i],c[i], c_prime[i]);
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
        output[i] = c_prime[i] + b[i] - r_mk2[i];
        /* val[i] = val[i] + T(1); */
    }
    delete[] r_mk2;
    delete[] r_msb;
    delete[] c_prime;
    delete[] b;
}

template<int rm = 0, int rk = BITLENGTH, typename Share, typename Datatype, typename FUNC_OP>
static void pack_additive(const Additive_Share<Datatype, Share>*  input, Additive_Share<Datatype, Share>*  output, const int len, FUNC_OP op){
    using sint = sint_t<Additive_Share<Datatype, Share>>;
    int m = len;
    sint* tmp = new sint[(m-1)/BITLENGTH+1];
    sint* tmp_output = new sint[(m-1)/BITLENGTH+1];
    int counter = 0;
    while(m > (BITLENGTH-1))
    {
       tmp[counter++] = sint::load_shares(input+counter*BITLENGTH);
       m -= BITLENGTH;
    }
    if(m > 0)
        tmp[counter++] = sint::load_shares(m, input+counter*BITLENGTH);
    op(tmp, tmp_output, counter);
    /* for(int i = 0; i < counter; i++) */
    /* { */
        /* std::cout << tmp[i].get_p1() << std::endl; */
    /* } */
    counter = 0;
    m = len;
    while(m > (BITLENGTH-1))
    {
        for(int j = 0; j < BITLENGTH; j++)
        {
            /* output[counter*BITLENGTH+j] = tmp[counter].get_share_pointer()[j]; */
            output[counter*BITLENGTH+j] = tmp_output[counter].get_share(j);
        }
        counter++;
        m -= BITLENGTH;
    }
    if(m > 0)
    {
        for(int j = 0; j < m; j++)
        {
            output[counter*BITLENGTH+j] = tmp_output[counter].get_share_pointer()[j];
        }
    }
    delete[] tmp;
    delete[] tmp_output;
}

template<typename Share>
void test_comp_trunc(DATATYPE *res)
{
    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    //Syntax for additive shares
    A* input = new A[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        if(i % 10 == 0)
            input[i] = PROMOTE(0);
        else if(i % 2 == 0)
            input[i] = PROMOTE(i);
        else
            input[i] = PROMOTE(-i);
    }
    A* ltz_output = new A[NUM_INPUTS];
    A* eqz_output = new A[NUM_INPUTS];
    pack_additive<0, BITLENGTH>(input, ltz_output, NUM_INPUTS, LTZ<0, BITLENGTH, Share, DATATYPE>); //LTZ
    pack_additive<0, BITLENGTH>(input, eqz_output, NUM_INPUTS, EQZ<0, BITLENGTH, Share, DATATYPE>); //EQZ
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        input[i].prepare_reveal_to_all();
        ltz_output[i].prepare_reveal_to_all();
        eqz_output[i].prepare_reveal_to_all();
    } 
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        auto inp = input[i].complete_reveal_to_all();
        auto res = ltz_output[i].complete_reveal_to_all();
        auto res2 = eqz_output[i].complete_reveal_to_all();
        UINT_TYPE uint_inp[DATTYPE/BITLENGTH];
        UINT_TYPE ltz_res[DATTYPE/BITLENGTH];
        UINT_TYPE eqz_res[DATTYPE/BITLENGTH];
        unorthogonalize_arithmetic(&inp, uint_inp, 1);
        unorthogonalize_arithmetic(&res, ltz_res, 1);
        unorthogonalize_arithmetic(&res2, eqz_res, 1);
        for(int j = 0; j < DATTYPE/BITLENGTH; j++)
        {
            std::cout << "Result: " << "input: " << INT_TYPE(uint_inp[j]) << " LTZ: " << ltz_res[j] << " EQZ: " << eqz_res[j] << std::endl;
        }

    }

    /* trunc_pr<A>(input, output, NUM_INPUTS); */
    delete[] input;
    delete[] ltz_output;
    delete[] eqz_output;

    //Syntax for sint
    /* sint* sint_input = new sint[NUM_INPUTS]; */
    /* sint* sint_output = new sint[NUM_INPUTS]; */
    /* LTZ<0, BITLENGTH, Share, DATATYPE>(sint_input, sint_output, NUM_INPUTS); */
    /* EQZ<0, BITLENGTH, Share, DATATYPE>(sint_input, sint_output, NUM_INPUTS); */
    /* trunc_pr<sint>(sint_input, sint_output, NUM_INPUTS); */
    /* delete[] sint_input; */
    /* delete[] sint_output; */
}


