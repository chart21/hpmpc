#pragma once
#include "../../protocols/Protocols.h"
#include <cstring>
#include <iostream>
#include "../../datatypes/k_bitset.hpp"

#define FUNCTION search
#define RESULTTYPE DATATYPE

    template<typename Pr, typename S>
void search(Pr P,/*outputs*/ DATATYPE *found)
{

// allocate memory for shares
DATATYPE (*dataset)[BITLENGTH][Pr::VALS_PER_SHARE] = (DATATYPE ((*)[BITLENGTH][Pr::VALS_PER_SHARE])) P.alloc_Share(((int) NUM_INPUTS)*BITLENGTH);
/* DATATYPE (*element)[Pr::VALS_PER_SHARE] = P.alloc_Share(BITLENGTH); */
DATATYPE (&element)[BITLENGTH][Pr::VALS_PER_SHARE] = P.alloc_Share(BITLENGTH);




/* P.prepare_receive_from( (*dataset)[Pr::VALS_PER_SHARE],P0,(NUM_INPUTS)*BITLENGTH, OP_ADD, OP_SUB); */
P.prepare_receive_from(element,P1,BITLENGTH, OP_ADD, OP_SUB);


P.communicate();



P.complete_receive_from((S*) dataset,P0,(NUM_INPUTS)*BITLENGTH, OP_ADD, OP_SUB);
P.complete_receive_from(element,P1,BITLENGTH, OP_ADD, OP_SUB);


for (int i = 0; i < NUM_INPUTS; i++) {
    for (int j = 0; j < BITLENGTH; j++) {
      dataset[i][j] = P.Not(P.Add(dataset[i][j], element[j], FUNC_XOR));
    }
  }
  
int c = 1;
  for (int k = BITLENGTH >> 1; k > 0; k = k >> 1) {
    for (int i = 0; i < k; i++) {
        int j = i * 2;
      for (int s = 0; s < NUM_INPUTS; s++) {
          P.prepare_mult(dataset[s][j],dataset[s][j +1], dataset[s][i], FUNC_XOR, FUNC_XOR, FUNC_AND);
      }
    }

    P.communicate(); 

    for (int i = 0; i < k; i++) {
        int j = i * 2;
      for (int s = 0; s < NUM_INPUTS; s++) {
            P.complete_mult(dataset[s][i], FUNC_XOR, FUNC_XOR);
      }
      


    }

    P.communicate();


  }
 
  *found = SET_ALL_ZERO(); 
  
  S sfound = dataset[0][0];

  for (int i = 1; i < NUM_INPUTS; i++) {
    sfound = P.Add(dataset[i][0],sfound, FUNC_XOR);

  }

P.prepare_reveal_to_all(sfound);
P.communicate();
*found = P.complete_Reveal(sfound,FUNC_XOR,FUNC_XOR);
P.communicate();

}

/* template<typename Pr, typename S> */
/* void search(Pr P, DATATYPE *found) */
/* { */

/* // allocate memory for shares */
/* S (*dataset)[BITLENGTH] = (S ((*)[BITLENGTH])) P.alloc_Share(((int) NUM_INPUTS)*BITLENGTH); */
/* S* element = P.alloc_Share(BITLENGTH); */



/* P.prepare_receive_from((S*) dataset,P0,(NUM_INPUTS)*BITLENGTH, OP_ADD, OP_SUB); */
/* P.prepare_receive_from(element,P1,BITLENGTH, OP_ADD, OP_SUB); */


/* P.communicate(); */



/* P.complete_receive_from((S*) dataset,P0,(NUM_INPUTS)*BITLENGTH, OP_ADD, OP_SUB); */
/* P.complete_receive_from(element,P1,BITLENGTH, OP_ADD, OP_SUB); */


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

void print_result(DATATYPE* var) 
{
    printf("P%i: Result: ", PARTY);
    uint8_t v8val[sizeof(DATATYPE)];
    std::memcpy(v8val, var, sizeof(v8val));
    for (uint i = sizeof(DATATYPE); i > 0; i--)
        std::cout << std::bitset<sizeof(uint8_t)*8>(v8val[i-1]);
    printf("\n");
}



