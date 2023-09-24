#pragma once
#include <iostream>
#include "../../protocols/Protocols.h"
#include "../../protocols/XOR_Share.hpp"
#include "../../protocols/Additive_Share.hpp"

#define FUNCTION debug
#define RESULTTYPE DATATYPE[num_players][BITLENGTH]

void compare(DATATYPE var[num_players][BITLENGTH], std::string test_func)
{
if (current_phase != 1)
    return;
auto inputs = new DATATYPE[num_players][BITLENGTH];
for(int i = 0; i < num_players; i++)
{
int num_erros = 0;
for(int j = 0; j < BITLENGTH; j++)
{

#if FUNCTION_IDENTIFIER == 7
/* inputs[i][j] = 3; */
/* if(3 != var[i][j] && 9 != var[i][j]) */
std::cout << "P" << PARTY << " " << var[i][j] << std::endl;
if(var[i][j] < 2 || var[i][j] > 242)
{
    num_erros++;
    std::cout << PARTY << " " << var[i][j] << " " << i << " " << j << std::endl;
}
#else
inputs[i][j] = SET_ALL_ONE();
if(j == i)
    inputs[i][j] = SET_ALL_ZERO();
if(inputs[i][j] != var[i][j])
{
    num_erros++;
    std::cout << "P" << PARTY << " " << var[i][j] << " " << i << " " << j << " " << inputs[i][j] << std::endl;
}
#endif
}
    std::cout << "P" << PARTY << ": " << num_erros << " Errors while testing " << test_func << ", input from player " << i << "\n";

}

}

template<typename Protocol>
void debug (/*outputs*/ DATATYPE result[num_players][BITLENGTH])
{
#if FUNCTION_IDENTIFIER == 7
using S = Additive_Share<DATATYPE, Protocol>;
#else
using S = XOR_Share<DATATYPE, Protocol>;
#endif
// allocate memory for shares
S (*inputs)[BITLENGTH] = new S[num_players][BITLENGTH];

/* if(player_id == 0) */
/* { */
/* /1* for (int i = 0; i < n; i++) { *1/ */
/* /1*     for (int j = 0; j < BITLENGTH; j++) { *1/ */
/* /1*       dataset[i][j] = P.share(dataset[i][j]); *1/ */
/* /1*   } *1/ */
/* /1* } *1/ */
/* /1* P_share( (DATATYPE*) dataset,n*BITLENGTH); *1/ */
/* P.share( (S*) dataset,(n)*BITLENGTH); */
/* } */
/* else if(player_id == 1) */
/* { */

/* P.share(element,BITLENGTH); */
/*     /1* for (int j = 0; j < BITLENGTH; j++) *1/ */ 
/*     /1*   element[j] = P.share(element[j]); *1/ */
/* /1* P_share(element,BITLENGTH); *1/ */
/* } */
for(int i = 0; i < BITLENGTH; i++)
{
inputs[0][i].template prepare_receive_from<P_0>();
inputs[1][i].template prepare_receive_from<P_1>();
inputs[2][i].template prepare_receive_from<P_2>();
#if num_players > 3
inputs[3][i].template prepare_receive_from<P_3>();
#endif
}
Protocol::communicate();

for(int i = 0; i < BITLENGTH; i++)
{
inputs[0][i].template complete_receive_from<P_0>();
inputs[1][i].template complete_receive_from<P_1>();
inputs[2][i].template complete_receive_from<P_2>();
#if num_players > 3
inputs[3][i].template complete_receive_from<P_3>();
#endif
}

Protocol::communicate();

for(int j = 0; j < num_players; j++)
    for (int i = 0; i < BITLENGTH; i++)
    inputs[j][i].prepare_reveal_to_all();
Protocol::communicate();

for(int j = 0; j < num_players; j++)
    for (int i = 0; i < BITLENGTH; i++)
        result[j][i] = inputs[j][i].complete_reveal_to_all();

Protocol::communicate();

for(int j = 0; j < num_players; j++)
    for (int i = 0; i < BITLENGTH; i++)
    inputs[j][i].prepare_reveal_to_all();
Protocol::communicate();

for(int j = 0; j < num_players; j++)
    for (int i = 0; i < BITLENGTH; i++)
        result[j][i] = inputs[j][i].complete_reveal_to_all();

Protocol::communicate();


std::cout <<   "P" << PARTY <<  ": ""Testing secret sharing and revealing: " << std::endl;
compare(result, "secret sharing and revealing");




for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {
#if FUNCTION_IDENTIFIER == 7
    inputs[j][i] = inputs[j][i] * inputs[j][i];
#else
    inputs[j][i] = inputs[j][i] & inputs[j][i];
#endif
}
}

Protocol::communicate();

for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {
#if FUNCTION_IDENTIFIER == 7
    inputs[j][i].complete_mult();
#else
    inputs[j][i].complete_and();
#endif
}
}
Protocol::communicate();
for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {

    inputs[j][i].prepare_reveal_to_all();
}
}
Protocol::communicate();

for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {

    result[j][i] = inputs[j][i].complete_reveal_to_all();
}
}
Protocol::communicate();
std::cout <<"P" << PARTY <<  ": " "Testing and/mult gates: " << std::endl;
compare(result, "and/mult gates");


for(int j = 0; j < num_players; j++)
{
for (int i = 0; i < BITLENGTH; i++) {
#if FUNCTION_IDENTIFIER == 7
    inputs[j][i] = inputs[j][i] + inputs[j][i];
#else
    inputs[j][i] = inputs[j][i] ^ inputs[j][i];
    if(i != j)
        inputs[j][i] = !inputs[j][i];
#endif
}
}

for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {
    inputs[j][i].prepare_reveal_to_all();
}

}

Protocol::communicate();


for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {

    result[j][i] = inputs[j][i].complete_reveal_to_all();
}
}

std::cout <<"P" << PARTY <<  ": " "Testing NOT, XOR/ADD gates: " << std::endl;
compare(result, "NOT, XOR/ADD gates");

}
// Reveal
//

/* void print_result(DATATYPE* var) */ 
/* { */
/*     uint8_t v8val[sizeof(DATATYPE)]; */
/*     std::memcpy(v8val, var, sizeof(v8val)); */
/*     for (uint i = sizeof(DATATYPE); i > 0; i--) */
/*         std::cout << std::bitset<sizeof(uint8_t)*8>(v8val[i-1]) << std::endl; */
/*         //std::cout << v8val[i]<< std::endl; */
/* } */


void print_result(DATATYPE var[][BITLENGTH]) 
{
}
    




