#pragma once
#include "../utils/xorshift.h"
#include "functions/search.hpp"

void print_bool(uint8_t* found)
{
    std::cout << "P" << PARTY << ": Expected Result: ";
    for (int j = 0; j < BITS_PER_REG; j++)
       std::cout << static_cast<int>(found[j]);
    std::cout << '\n';
}

void search_Compare(UINT_TYPE origData[NUM_INPUTS][DATTYPE], UINT_TYPE origElements[], uint8_t* found)
{
 /* for (int i = 0; i < NUM_INPUTS; i++) { */
 /*    for (int j = 0; j < BITS_PER_REG; j++) { */
 /*        if(origData[i][j] == origElements[j]) */
 /*            found[j] = 1; */
 /*    } */   
/* } */
    if(DATTYPE > 7)
        found[7] = 1;
}


void insertManually(DATATYPE dataset[NUM_INPUTS][BITLENGTH], DATATYPE elements[NUM_INPUTS], UINT_TYPE origData[NUM_INPUTS][DATTYPE], UINT_TYPE origElements[], int c, int b, uint64_t numElement, uint64_t numDataset ){

/* unorthogonalize(elements, origElements); */

/* for (int i = 0; i < NUM_INPUTS; i++) { */
/*  unorthogonalize(dataset[i], origData[i]); */   
/* } */
origData[c][b] = numDataset;
origElements[b] = numElement;
/* std::cout << origData[c][b] << origElements[b] << std::endl; */


uint8_t* cfound = new uint8_t[BITS_PER_REG]{0};
/* funcTime("Plain search", search_Compare, origData, origElements, cfound); */

#if PRINT == 1
search_Compare(origData, origElements, cfound);
print_bool(cfound);
#endif



orthogonalize_boolean(origElements, elements);

for (int i = 0; i < NUM_INPUTS; i++) {
 orthogonalize_boolean(origData[i], dataset[i]);   
}
}
void randomizeInputs(DATATYPE dataset[NUM_INPUTS][BITLENGTH],
                     DATATYPE elements[NUM_INPUTS]) {
// init randomization
UINT_TYPE *iseed = NEW(UINT_TYPE[BITS_PER_REG]);
for (int i = 0; i < BITS_PER_REG; i++) {
 iseed[i] = rand();
}
DATATYPE *seed = new DATATYPE[BITLENGTH];
orthogonalize_boolean(iseed, seed);

// generate random data
for (int i = 0; i < int(NUM_INPUTS / (sizeof(DATATYPE) * 8) / BITS_PER_REG);
     i++) {
 xor_shift__(seed, dataset[i]);
}
xor_shift__(seed, elements);
/* funcTime("xor_shift",xor_shift__,seed, elements); */
}

//init 1 srng per link

void init_comm() {

// sharing
input_length[0] = BITLENGTH * NUM_INPUTS;
input_length[1] = BITLENGTH;

// revealing
reveal_length[0] = 1;
reveal_length[1] = 1;
reveal_length[2] = 1;

total_comm = 2 - use_srng_for_inputs;

// function

for (int k = BITLENGTH >> 1; k > 0; k = k >> 1) {
 total_comm = total_comm + 1;
}
elements_per_round = new int[total_comm];

// function

int i = 1 - use_srng_for_inputs;
for (int k = BITLENGTH >> 1; k > 0; k = k >> 1) {
 elements_per_round[i] = k * NUM_INPUTS;
 i += 1;
}
}

void generateElements()
{
UINT_TYPE (*origData)[DATTYPE] = NEW(UINT_TYPE[NUM_INPUTS][DATTYPE]);
UINT_TYPE *origElements = NEW(UINT_TYPE[DATTYPE]);

/* DATATYPE (*dataset)[BITLENGTH] = NEW(DATATYPE[NUM_INPUTS][BITLENGTH]); */
/* DATATYPE* elements = NEW(DATATYPE[BITLENGTH]); */
/* for (int i = 0; i < NUM_INPUTS; i++) { */
/*  orthogonalize(origData[i], dataset[i]); */   
/* } */
/* orthogonalize(origElements, elements); */

// generate random input data instead of reading from file
/* funcTime("generating random inputs",randomizeInputs,dataset,elements); */

/* DATATYPE (*dataset)[BITLENGTH] = ((DATATYPE(*)[BITLENGTH]) origData); */ 
/* DATATYPE* elements = ((DATATYPE*) origElements); */


randomizeInputs( (DATATYPE(*)[BITLENGTH]) origData, (DATATYPE*)  origElements);

//modify certain data to test functionality
//
DATATYPE (*dataset)[BITLENGTH] = NEW(DATATYPE[NUM_INPUTS][BITLENGTH]);
DATATYPE* elements = NEW(DATATYPE[BITLENGTH]);


insertManually(dataset, elements, origData, origElements, 1,7 , 200, 200);

if(player_id == 0)
{
    player_input = (DATATYPE*) dataset;
}
else if(player_id == 1)
    player_input = elements;

}

