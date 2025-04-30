#pragma once
#include "../../datatypes/k_bitset.hpp"
#include "../../protocols/Protocols.h"
#include "../../protocols/XOR_Share.hpp"
#define FUNCTION search
#define RESULTTYPE DATATYPE

void print_result(DATATYPE* var)
{
    printf("P%i: Result: ", PARTY);
    uint8_t v8val[sizeof(DATATYPE)];
    std::memcpy(v8val, var, sizeof(v8val));
    for (uint i = sizeof(DATATYPE); i > 0; i--)
        std::cout << std::bitset<sizeof(uint8_t) * 8>(v8val[i - 1]);
    printf("\n");
}

template <typename Protocol>
void search(/*outputs*/ DATATYPE* found)
{
    using S = XOR_Share<DATATYPE, Protocol>;
    using Bitset = sbitset_t<BITLENGTH, S>;

    /* S (*dataset)[BITLENGTH] = new S [NUM_INPUTS][BITLENGTH]; */
    /* S *element = new S[BITLENGTH]; */
    Bitset* dataset = new Bitset[NUM_INPUTS];
    Bitset element;

    /* Share (*dataset)[BITLENGTH] = (Share ((*)[BITLENGTH])) new Share[((int) NUM_INPUTS)*BITLENGTH]; */
    /* Share* element = new Share[BITLENGTH]; */

    for (int i = 0; i < NUM_INPUTS; i++)
        dataset[i].template prepare_receive_from<P_0>();

    element.template prepare_receive_from<P_1>();

    Protocol::communicate();

    for (int i = 0; i < NUM_INPUTS; i++)
        dataset[i].template complete_receive_from<P_0>();

    element.template complete_receive_from<P_1>();

    for (int i = 0; i < NUM_INPUTS; i++)
    {
        dataset[i] = ~(dataset[i] ^ element);
    }

    for (int k = BITLENGTH >> 1; k > 0; k = k >> 1)
    {
        for (int i = 0; i < k; i++)
        {
            int j = i * 2;
            for (int s = 0; s < NUM_INPUTS; s++)
            {
                dataset[s][i] = dataset[s][j] & dataset[s][j + 1];
            }
        }

        Protocol::communicate();

        for (int i = 0; i < k; i++)
        {
            for (int s = 0; s < NUM_INPUTS; s++)
            {
                dataset[s][i].complete_and();
            }
        }

        Protocol::communicate();
    }

    for (int i = 1; i < NUM_INPUTS; i++)
    {
        dataset[0][0] = dataset[i][0] ^ dataset[0][0];
    }

    dataset[0][0].prepare_reveal_to_all();

    Protocol::communicate();

    *found = dataset[0][0].complete_reveal_to_all();

    Protocol::communicate();
}
