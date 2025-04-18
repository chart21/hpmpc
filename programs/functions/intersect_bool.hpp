#pragma once
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
template <typename Protocol>
void intersect_bool(const sbitset_t<BITLENGTH, XOR_Share<DATATYPE, Protocol>>* a,
                    const sbitset_t<BITLENGTH, XOR_Share<DATATYPE, Protocol>>* b,
                    sbitset_t<BITLENGTH, XOR_Share<DATATYPE, Protocol>>* result,
                    const int len_a,
                    const int len_b)
{
    using S = XOR_Share<DATATYPE, Protocol>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    auto tmp = new Bitset[len_a * len_b];

    assert(len_a <= len_b);
    for (int i = 0; i < len_a; i++)
        for (int j = 0; j < len_b; j++)
            tmp[i * len_b + j] = ~(a[i] ^ b[j]);  // vals[i] == element

    for (int k = BITLENGTH >> 1; k > 0; k = k >> 1)
    {
        for (int i = 0; i < k; i++)
        {
            int j = i * 2;
            for (int s = 0; s < len_a; s++)
            {
                for (int t = 0; t < len_b; t++)
                    tmp[s * len_b + t][i] =
                        tmp[s * len_b + t][j] &
                        tmp[s * len_b + t][j + 1];  // Only if all bits of the comparison are 1, the result should be 1
            }
        }

        Protocol::communicate();

        for (int i = 0; i < k; i++)
        {
            for (int s = 0; s < len_a; s++)
            {
                for (int t = 0; t < len_b; t++)
                    tmp[s * len_b + t][i].complete_and();
            }
        }

        Protocol::communicate();
    }

    auto intersect = new S[len_a];

    for (int i = 0; i < len_a; i++)
    {
        intersect[i] = SET_ALL_ZERO();
        for (int j = 1; j < len_b; j++)
            intersect[i] =
                intersect[i] ^
                tmp[i * len_b + j]
                   [0];  // intersect is 1 if element has been found in any of the comparisons (assumes no duplicates)
    }

    for (int i = 0; i < len_a; i++)
        for (int k = 0; k < BITLENGTH; k++)
            result[i][k] = a[i][k] & intersect[i];  // store element as a result if it has been found, otherwise 0
    Protocol::communicate();
    for (int i = 0; i < len_a; i++)
        for (int k = 0; k < BITLENGTH; k++)
            result[i][k].complete_and();

    delete[] tmp;
    delete[] intersect;
}
