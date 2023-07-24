#pragma once
#include <iostream>
#include "../../protocols/Protocols.h"

#define FUNCTION debug
#define RESULTTYPE DATATYPE[num_players][BITLENGTH]

void compare(DATATYPE var[num_players][BITLENGTH]) 
{
auto inputs = new DATATYPE[num_players][BITLENGTH];
for(int i = 0; i < num_players; i++)
{
int num_erros = 0;
for(int j = 0; j < BITLENGTH; j++)
{

#if FUNCTION_IDENTIFIER == 7
inputs[i][j] = 3;
if(3 != var[i][j] && 9 != var[i][j])
{
    num_erros++;
    std::cout << var[i][j] << " " << i << " " << j << std::endl;
}
#else
inputs[i][j] = SET_ALL_ONE();
if(j == i)
    inputs[i][j] = SET_ALL_ZERO();
if(inputs[i][j] != var[i][j])
    num_erros++;
#endif
}
    std::cout << num_erros << " Errors in compare, input from player " << i << std::endl;

}

}

template<typename Pr, typename S>
void debug (Pr P,/*outputs*/ DATATYPE result[num_players][BITLENGTH])
{

// allocate memory for shares
S (*inputs)[BITLENGTH] = (S ((*)[BITLENGTH])) P.alloc_Share(((int) num_players)*BITLENGTH);

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
P.prepare_receive_from(inputs[0] ,P0,BITLENGTH, OP_ADD, OP_SUB);
P.prepare_receive_from(inputs[1] ,P1,BITLENGTH, OP_ADD, OP_SUB);
P.prepare_receive_from(inputs[2] ,P2,BITLENGTH, OP_ADD, OP_SUB);
#if num_players > 3
P.prepare_receive_from(inputs[3] ,P3,BITLENGTH, OP_ADD, OP_SUB);
#endif

P.communicate();


P.complete_receive_from(inputs[0] ,P0,BITLENGTH, OP_ADD, OP_SUB);
P.complete_receive_from(inputs[1] ,P1,BITLENGTH, OP_ADD, OP_SUB);
P.complete_receive_from(inputs[2] ,P2,BITLENGTH, OP_ADD, OP_SUB);
#if num_players > 3
P.complete_receive_from(inputs[3] ,P3,BITLENGTH, OP_ADD, OP_SUB);
#endif

P.communicate();

for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {

    P.prepare_reveal_to_all(inputs[j][i]);
}
}
P.communicate();

for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {

    result[j][i] = P.complete_Reveal(inputs[j][i], OP_ADD, OP_SUB);
}
}

P.communicate();


std::cout << "Testing secret sharing and revealing: " << std::endl;
compare(result);



for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {
    P.prepare_mult(inputs[j][i],inputs[j][i],inputs[j][i],OP_ADD, OP_SUB, OP_MULT);
}
}
P.communicate();
for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {
    P.complete_mult(inputs[j][i], OP_ADD, OP_SUB);
}
}

for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {

    P.prepare_reveal_to_all(inputs[j][i]);
}
}
P.communicate();

for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {

    result[j][i] = P.complete_Reveal(inputs[j][i], OP_ADD, OP_SUB);
}
}

std::cout << "Testing and gates: " << std::endl;
compare(result);

#if FUNCTION_IDENTIFIER != 7

for(int j = 0; j < num_players; j++)
{
for (int i = 0; i < BITLENGTH; i++) {

    inputs[j][i] = P.Add(inputs[j][i],inputs[j][i],OP_ADD);
    if(i != j)
        inputs[j][i] = P.Not(inputs[j][i]);
}
}

for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {

    P.prepare_reveal_to_all(inputs[j][i]);
}
}
P.communicate();

for(int j = 0; j < num_players; j++)
{

for (int i = 0; i < BITLENGTH; i++) {

    result[j][i] = P.complete_Reveal(inputs[j][i], OP_ADD, OP_SUB);
}
}

std::cout << "Testing NOT, XOR gates: " << std::endl;
compare(result);
#endif

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
    




