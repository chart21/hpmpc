#pragma once
#include "../../protocols/Protocols.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <bitset>
#include "../../protocols/XOR_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include "boolean_adder.hpp"
#define FUNCTION convert_share
#define RESULTTYPE DATATYPE

template<typename Share>
void convert_share(/*outputs*/ DATATYPE *result)
{
    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<S>;
    using sint = sint_t<S>;

    sint val;
    Bitset y;
    val.template prepare_receive_from<P0>();
    Share::communicate();
    val.template complete_receive_from<P0>();
    Share::communicate();
    Bitset s1 = sbitset_t<S>::prepare_A2B_S1(val.get_share_pointer());
    Bitset s2 = sbitset_t<S>::prepare_A2B_S2(val.get_share_pointer());
    Share::communicate();
    s1.complete_A2B_S1();
    s2.complete_A2B_S2();
    BooleanAdder<S> adder(s1, s2,y);
    while(!adder.is_done())
    {
        adder.step();
        Share::communicate();
    }
    /* y = val; */
    y.prepare_reveal_to_all();
    /* s1.prepare_reveal_to_all(); */
    /* s2.prepare_reveal_to_all(); */
    Share::communicate();
    uint64_t result_arr[DATTYPE];
    /* uint64_t s1_arr[DATTYPE]; */
    /* uint64_t s2_arr[DATTYPE]; */
    /* s1.complete_reveal_to_all(s1_arr); */
    /* s2.complete_reveal_to_all(s2_arr); */

    y.complete_reveal_to_all(result_arr);
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
