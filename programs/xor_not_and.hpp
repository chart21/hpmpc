#pragma once
#include <iostream>
#include <cstring>
#include "functions/xor_not_and.hpp"

#define FUNCTION XORNOTAND

void print_bool(uint8_t* found)
{
     for (int j = 0; j < BITS_PER_REG; j++)
         std::cout << +found[j];
    std::cout << '\n';
}


void compare(UINT_TYPE origData[num_players][DATTYPE], UINT_TYPE result[DATTYPE])
{
    for (int j = 0; j < DATTYPE; j++) {
        origData[0][j] = !( origData[0][j] ^ origData[1][j] );
#if num_players == 4
        origData[2][j] = !( origData[2][j] ^ origData[3][j] );
#endif
        origData[0][j] = origData[0][j] && origData[2][j];

        result[j] = origData[0][j];
    }   
}






void generateElements()
{


DATATYPE (*dataset)[BITLENGTH] = NEW(DATATYPE[num_players][BITLENGTH]);


for (int i = 0; i < num_players; i++) {
    for (int j = 0; j < BITLENGTH; j++) {
        dataset[i][j] = SET_ALL_ONE();
    }
}

player_input = dataset[PARTY];

alignas(DATTYPE) UINT_TYPE regular_data[num_players][DATTYPE];
alignas(DATTYPE) UINT_TYPE result[DATTYPE];
for (int i = 0; i < num_players; i++) {
unorthogonalize_boolean(dataset[i], regular_data[i]);
    }
    //
#if PRINT == 1
print_result(dataset[0]);
compare(regular_data, result);
print_bool((uint8_t*)result);
#endif
}




