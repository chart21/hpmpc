#pragma once
#include "../../protocols/Protocols.h"
#include <cstring>
#include <iostream>
#include <bitset>


#define FUNCTION search
#define RESULTTYPE DATATYPE

template<typename Pr, typename S>
void search(Pr P,/*outputs*/ DATATYPE *found)
{

// allocate memory for shares
S (*dataset)[BITLENGTH] = (S ((*)[BITLENGTH])) P.alloc_Share(((int) NUM_INPUTS)*BITLENGTH);
S* element = P.alloc_Share(BITLENGTH);



P.prepare_receive_from((S*) dataset,P0,(NUM_INPUTS)*BITLENGTH);
P.prepare_receive_from(element,P1,BITLENGTH);


P.communicate();



P.complete_receive_from((S*) dataset,P0,(NUM_INPUTS)*BITLENGTH);
P.complete_receive_from(element,P1,BITLENGTH);


for (int i = 0; i < NUM_INPUTS; i++) {
    for (int j = 0; j < BITLENGTH; j++) {
      dataset[i][j] = P.Not(P.Xor(dataset[i][j], element[j]));
    }
  }
  
int c = 1;
  for (int k = BITLENGTH >> 1; k > 0; k = k >> 1) {
    for (int i = 0; i < k; i++) {
        int j = i * 2;
      for (int s = 0; s < NUM_INPUTS; s++) {
          P.prepare_and(dataset[s][j],dataset[s][j +1], dataset[s][i]);
      }
    }

    P.communicate(); 

    for (int i = 0; i < k; i++) {
        int j = i * 2;
      for (int s = 0; s < NUM_INPUTS; s++) {
            P.complete_and(dataset[s][i]);
      }
      


    }

    P.communicate();


  }
 
  *found = SET_ALL_ZERO(); 
  
  S sfound = dataset[0][0];

  for (int i = 1; i < NUM_INPUTS; i++) {
    sfound = P.Xor(dataset[i][0],sfound); 

  }

P.prepare_reveal_to_all(sfound);
P.communicate();
*found = P.complete_Reveal(sfound);
P.communicate();

}
// Reveal
//

void print_result(DATATYPE* var) 
{
    printf("P%i: Result: ", PARTY);
    uint8_t v8val[sizeof(DATATYPE)];
    std::memcpy(v8val, var, sizeof(v8val));
    for (uint i = sizeof(DATATYPE); i > 0; i--)
        std::cout << std::bitset<sizeof(uint8_t)*8>(v8val[i-1]);
    printf("\n");
}



