#pragma once
#include "../../protocols/Protocols.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <bitset>
#include "../../protocols/XOR_Share.hpp"
#include "../../protocols/Additive_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
/* #include "boolean_adder.hpp" */
#include "boolean_adder_updated.hpp"

#include "boolean_adder_msb.hpp"
#include "ppa_msb.hpp"
#include "ppa.hpp"
#include "ppa_msb_unsafe.hpp"

/* #include "boolean_adder.hpp" */
/* #include "ppa.hpp" */
#define FUNCTION convert_share
#define RESULTTYPE DATATYPE
    template<typename Share>
void adder(DATATYPE* res)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<S>;
    using sint = sint_t<A>;
    
    Bitset x;
    Bitset y;
    x.template prepare_receive_from<P_0>();
    y.template prepare_receive_from<P_0>();
    Share::communicate();
    x.template complete_receive_from<P_0>();
    y.template complete_receive_from<P_0>();
    Share::communicate();
    Bitset z;
    BooleanAdder<S> adder(x, y, z);
    while(!adder.is_done())
    {
        adder.step();
        Share::communicate();
    }
    z.prepare_reveal_to_all();
    Share::communicate();
    uint64_t result_arr[DATTYPE];

    z.complete_reveal_to_all(result_arr);
    if(current_phase == 1)
    {
        std::cout << "P" << PARTY << ": Result: ";
    for(int i = 0; i < DATTYPE; i++)
    {
        /* std::cout << std::bitset<sizeof(uint64_t)*8>(s1_arr[i] + s2_arr[i]); */
    /* std::cout << std::endl; */
        std::cout << std::bitset<sizeof(uint64_t)*8>(result_arr[i]);
    std::cout << std::endl;
    }

}
}
    template<typename Share>
void RELU(DATATYPE* res)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<S>;
    using sint = sint_t<A>;
    
    sint* val = new sint[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        val[i].template prepare_receive_from<P_0>();
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        val[i].template complete_receive_from<P_0>();
    }
    Share::communicate();
    Bitset *s1 = new Bitset[NUM_INPUTS];
    Bitset *s2 = new Bitset[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        s1[i] = sbitset_t<S>::prepare_A2B_S1( (S*) val[i].get_share_pointer());
        s2[i] = sbitset_t<S>::prepare_A2B_S2( (S*) val[i].get_share_pointer());
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        s1[i].complete_A2B_S1();
        s2[i].complete_A2B_S2();
    }
    Bitset* y = new Bitset[NUM_INPUTS];
    /* BooleanAdder<S> *adder = new BooleanAdder<S>[NUM_INPUTS]; */
    std::vector<PPA_MSB_Unsafe<S>> adders;
    adders.reserve(NUM_INPUTS);
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], y[i][0]);
    }
    while(!adders[0].is_done())
    {
        for(int i = 0; i < NUM_INPUTS; i++)
        {
            adders[i].step();
        }
        Share::communicate();
    }
    delete[] s1;
    delete[] s2;
    adders.clear();
    adders.shrink_to_fit();
    
    S *msb = new S[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        msb[i] = ~ y[i][0];
    }
    sint* t1 = new sint[NUM_INPUTS];
    sint* t2 = new sint[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        msb[i].prepare_bit_injection_S1(t1[i].get_share_pointer());
        msb[i].prepare_bit_injection_S2(t2[i].get_share_pointer());
    }
    delete[] msb;
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        t1[i].complete_bit_injection_S1();
        t2[i].complete_bit_injection_S2();
    }
    sint* result = new sint[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        result[i].prepare_XOR(t1[i],t2[i]);
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        result[i].complete_XOR(t1[i],t2[i]);
    }
    delete[] t1;
    delete[] t2;

    for(int i = 0; i < NUM_INPUTS; i++)
    {
        result[i] = result[i] * val[i];
    }
    delete[] val;
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        result[i].complete_mult();
    }


    for(int i = 0; i < NUM_INPUTS; i++)
    {
        result[i].prepare_reveal_to_all();
    }
    Share::communicate();
    auto result_arr = new UINT_TYPE[NUM_INPUTS][DATTYPE];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        result[i].complete_reveal_to_all(result_arr[i]);
    }
    delete[] result;
#if PRINT == 1
    if(current_phase == 1)
    {
        std::cout << "P" << PARTY << ": Result: ";
        for(int i = 0; i < NUM_INPUTS; i++)
        {
    for(int j = 0; j < DATTYPE; j++)
    {
        std::cout << std::bitset<sizeof(uint64_t)*8>(result_arr[i][j]);
    std::cout << std::endl;
    }
    std::cout << std::endl;
        }
    }
#endif

}


    template<typename Share>
void bit_injection(DATATYPE* res)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<S>;
    using sint = sint_t<A>;

    Bitset val;
    val.template prepare_receive_from<P_0>();
    Share::communicate();
    val.template complete_receive_from<P_0>();
    Share::communicate();
    S s = val[0];
    
    sint s1;
    sint s2;
    s.prepare_bit_injection_S1(s1.get_share_pointer());
    s.prepare_bit_injection_S2(s2.get_share_pointer());
    Share::communicate();
    s1.complete_bit_injection_S1();
    s2.complete_bit_injection_S2();
    sint result;
    result.prepare_XOR(s1,s2);
    Share::communicate();
    result.complete_XOR(s1,s2);

    /* Bitset result = val; */
    result.prepare_reveal_to_all();
    Share::communicate();
    uint64_t result_arr[DATTYPE];
    result.complete_reveal_to_all(result_arr);
    if(current_phase == 1)
    {
        std::cout << "P" << PARTY << ": Result: ";
    for(int i = 0; i < DATTYPE; i++)
    {
        /* std::cout << std::bitset<sizeof(uint64_t)*8>(s1_arr[i] + s2_arr[i]); */
    /* std::cout << std::endl; */
        std::cout << std::bitset<sizeof(uint64_t)*8>(result_arr[i]);
    std::cout << std::endl;
    }

}

}
template<typename Share>
void convert_share(/*outputs*/ DATATYPE *result)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<S>;
    using sint = sint_t<A>;

    sint val;
    Bitset y;
    /* Bitset val; */
    /* sint y; */
    val.template prepare_receive_from<P_0>();
    Share::communicate();
    val.template complete_receive_from<P_0>();
    Share::communicate();
    Bitset s1 = sbitset_t<S>::prepare_A2B_S1( (S*) val.get_share_pointer());
    Bitset s2 = sbitset_t<S>::prepare_A2B_S2( (S*) val.get_share_pointer());
    Share::communicate();
    s1.complete_A2B_S1();
    s2.complete_A2B_S2();
    BooleanAdder<S> adder(s1, s2,y);
    while(!adder.is_done())
    {
        adder.step();
        Share::communicate();
    }
    /* val = val + val; */
    
    /* y = val; */
    y.prepare_reveal_to_all();
    /* s1.prepare_reveal_to_all(); */
    /* s2.prepare_reveal_to_all(); */
    Share::communicate();
    auto result_arr = NEW(UINT_TYPE[DATTYPE]);
    /* uint64_t s1_arr[DATTYPE]; */
    /* uint64_t s2_arr[DATTYPE]; */
    /* s1.complete_reveal_to_all(s1_arr); */
    /* s2.complete_reveal_to_all(s2_arr); */

    y.complete_reveal_to_all(result_arr);
    /* DATATYPE temp[DATTYPE]; */
    /* orthogonalize_boolean(result_arr, temp); */
    /* temp[0] = ~ temp[0]; */
    /* temp[0] = NOT(temp[0]); */
    /* unorthogonalize_boolean(  temp, result_arr); */
    if(current_phase == 1)
    {
        std::cout << "P" << PARTY << ": Result: ";
    for(int i = 0; i < DATTYPE; i++)
    {
        /* std::cout << std::bitset<sizeof(uint64_t)*8>(s1_arr[i] + s2_arr[i]); */
    /* std::cout << std::endl; */
        std::cout << std::bitset<sizeof(UINT_TYPE)*8>(result_arr[i]);
    std::cout << std::endl;
    }
    

}
}
