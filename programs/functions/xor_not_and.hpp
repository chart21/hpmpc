#pragma once
#include <iostream>
#include <bitset>
#include <cstring>
#include "../../protocols/Protocols.h"
#define RESULTTYPE DATATYPE [ BITLENGTH ]

template<typename Pr, typename S>
void XORNOTAND (Pr P,/*outputs*/ DATATYPE result[])
{

// allocate memory for shares
S (*data)[BITLENGTH] = (S ((*)[BITLENGTH])) P.alloc_Share(((int) num_players)*BITLENGTH);




for (int i = 0; i < num_players; i++) {
P.prepare_receive_from(data[i],i,BITLENGTH,OP_ADD,OP_SUB);
  }


P.communicate();

// change to receive from
for (int i = 0; i < num_players; i++) {
P.complete_receive_from(data[i],i,BITLENGTH,OP_ADD,OP_SUB);
  }


    for (int j = 0; j < BITLENGTH; j++) {

 // XOR 
data[0][j] = P.Add(data[0][j],data[1][j], OP_ADD);


// 4 Players: XOR, 3 Players: NOT
#if num_players == 4
data[2][j] = P.Add(data[2][j],data[3][j], OP_ADD);
#else
data[2][j] = P.Not(data[2][j]);
#endif


// NOT
data[0][j] = P.Not(data[0][j]);
data[2][j] = P.Not(data[2][j]);


// AND
P.prepare_mult(data[0][j],data[2][j],data[0][j], OP_ADD, OP_SUB, OP_MULT);
    
}

  

P.communicate();

for (int j = 0; j < BITLENGTH; j++) {
P.complete_mult(data[0][j], OP_ADD, OP_SUB);
P.prepare_reveal_to_all(data[0][j]);
  }

P.communicate();

for (int j = 0; j < BITLENGTH; j++) {
result[j] = SET_ALL_ZERO(); 
result[j] = P.complete_Reveal(data[0][j], OP_ADD, OP_SUB);
  }

}
  
void print_result(DATATYPE result[]) 
{
    printf("P%i: Result ", PARTY);
    auto var = result[0];
    uint8_t v8val[sizeof(DATATYPE)];
    std::memcpy(v8val, &var, sizeof(v8val));
    for (uint i = sizeof(DATATYPE); i > 0; i--)
        std::cout << std::bitset<sizeof(uint8_t)*8>(v8val[i-1]);
    printf("\n");
}
     
