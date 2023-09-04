#pragma once
#include "../../protocols/Protocols.h"
#include <cstring>
#include <iostream>
#include <bitset>
#include "../../protocols/XOR_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#define FUNCTION search
#define RESULTTYPE DATATYPE

void print_result(DATATYPE* var) 
{
    printf("P%i: Result: ", PARTY);
    uint8_t v8val[sizeof(DATATYPE)];
    std::memcpy(v8val, var, sizeof(v8val));
    for (uint i = sizeof(DATATYPE); i > 0; i--)
        std::cout << std::bitset<sizeof(uint8_t)*8>(v8val[i-1]);
    printf("\n");
}


    template<typename Protocol>
void search(/*outputs*/ DATATYPE *found)
{
    using S = XOR_Share<DATATYPE, Protocol>;
    using Bitset = sbitset_t<S>;

    /* S (*dataset)[BITLENGTH] = new S [NUM_INPUTS][BITLENGTH]; */
    /* S *element = new S[BITLENGTH]; */
    Bitset* dataset = new Bitset[NUM_INPUTS];
    Bitset element;

/* Share (*dataset)[BITLENGTH] = (Share ((*)[BITLENGTH])) new Share[((int) NUM_INPUTS)*BITLENGTH]; */
/* Share* element = new Share[BITLENGTH]; */


for( int i = 0; i < NUM_INPUTS; i++)
    dataset[i].template prepare_receive_from<P_0>();

element.template prepare_receive_from<P_1>();


Protocol::communicate();

for( int i = 0; i < NUM_INPUTS; i++)
    dataset[i].template complete_receive_from<P_0>();

element.template complete_receive_from<P_1>();


for (int i = 0; i < NUM_INPUTS; i++) {
      dataset[i] = ~ (dataset[i] ^ element);
      
  }
  
  for (int k = BITLENGTH >> 1; k > 0; k = k >> 1) {
    for (int i = 0; i < k; i++) {
        int j = i * 2;
      for (int s = 0; s < NUM_INPUTS; s++) {
          dataset[s][i] = dataset[s][j] & dataset[s][j +1];
      }
    }

    Protocol::communicate(); 

    for (int i = 0; i < k; i++) {
      for (int s = 0; s < NUM_INPUTS; s++) {
          dataset[s][i].complete_and();
      }
      


    }

    Protocol::communicate();

  }
 
  /* *found = SET_ALL_ZERO(); */ 
  
/* S sfound = dataset[0][0]; */

  for (int i = 1; i < NUM_INPUTS; i++) {
    dataset[0][0] = dataset[i][0] ^ dataset[0][0];

  }


dataset[0][0].prepare_reveal_to_all();

Protocol::communicate();

*found = dataset[0][0].complete_reveal_to_all();

Protocol::communicate();

}
/* template<typename Pr, typename S> */
/* void search(Pr P, DATATYPE *found) */
/* { */

/* // allocate memory for shares */
/* S (*dataset)[BITLENGTH] = (S ((*)[BITLENGTH])) P.alloc_Share(((int) NUM_INPUTS)*BITLENGTH); */
/* S* element = P.alloc_Share(BITLENGTH); */



/* P.prepare_receive_from((S*) dataset,P_0,(NUM_INPUTS)*BITLENGTH, OP_ADD, OP_SUB); */
/* P.prepare_receive_from(element,P_1,BITLENGTH, OP_ADD, OP_SUB); */


/* P.communicate(); */



/* P.complete_receive_from((S*) dataset,P_0,(NUM_INPUTS)*BITLENGTH, OP_ADD, OP_SUB); */
/* P.complete_receive_from(element,P_1,BITLENGTH, OP_ADD, OP_SUB); */


/* for (int i = 0; i < NUM_INPUTS; i++) { */
/*     for (int j = 0; j < BITLENGTH; j++) { */
/*       dataset[i][j] = P.Not(P.Add(dataset[i][j], element[j], FUNC_XOR)); */
/*     } */
/*   } */
  
/* int c = 1; */
/*   for (int k = BITLENGTH >> 1; k > 0; k = k >> 1) { */
/*     for (int i = 0; i < k; i++) { */
/*         int j = i * 2; */
/*       for (int s = 0; s < NUM_INPUTS; s++) { */
/*           P.prepare_mult(dataset[s][j],dataset[s][j +1], dataset[s][i], FUNC_XOR, FUNC_XOR, FUNC_AND); */
/*       } */
/*     } */

/*     P.communicate(); */ 

/*     for (int i = 0; i < k; i++) { */
/*         int j = i * 2; */
/*       for (int s = 0; s < NUM_INPUTS; s++) { */
/*             P.complete_mult(dataset[s][i], FUNC_XOR, FUNC_XOR); */
/*       } */
      


/*     } */

/*     P.communicate(); */


/*   } */
 
/*   *found = SET_ALL_ZERO(); */ 
  
/*   S sfound = dataset[0][0]; */

/*   for (int i = 1; i < NUM_INPUTS; i++) { */
/*     sfound = P.Add(dataset[i][0],sfound, FUNC_XOR); */

/*   } */

/* P.prepare_reveal_to_all(sfound); */
/* P.communicate(); */
/* *found = P.complete_Reveal(sfound,FUNC_XOR,FUNC_XOR); */
/* P.communicate(); */

/* } */
// Reveal
//




