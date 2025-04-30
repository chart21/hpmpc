#pragma once
#include "../../datatypes/Additive_Share.hpp"
template <typename T>
static void trunc_2k_in_place(T* val, const int len, bool isPositive = false, int fractional_bits = FRACTIONAL)
{

#if MSB0_OPT == 1
    if (!isPositive)
#endif
        for (int i = 0; i < len; i++)
            val[i] = val[i] + T((UINT_TYPE(1) << (BITLENGTH - 1)));  // add 2^l-1 to gurantee positive number
    T* r_msb = new T[len];
    T* r_mk2 = new T[len];
    T* c = new T[len];
    T* c_prime = new T[len];
    T::communicate();
    for (int i = 0; i < len; i++)
    {
        val[i].prepare_trunc_2k_inputs(r_mk2[i], r_msb[i], c[i], c_prime[i], fractional_bits);
    }
    T::communicate();
    for (int i = 0; i < len; i++)
    {
        val[i].complete_trunc_2k_inputs(r_mk2[i], r_msb[i], c[i], c_prime[i]);
    }
    T::communicate();
    T* b = new T[len];
    for (int i = 0; i < len; i++)
        b[i].prepare_XOR(r_msb[i], c[i]);
    T::communicate();
    for (int i = 0; i < len; i++)
    {
        b[i].complete_XOR(r_msb[i], c[i]);
        b[i] = b[i].mult_public(UINT_TYPE(1) << (BITLENGTH - fractional_bits - 1));
    }
    T::communicate();
    delete[] c;

    for (int i = 0; i < len; i++)
    {
        val[i] = c_prime[i] + b[i] - r_mk2[i];
    }

#if MSB0_OPT == 1
    if (!isPositive)
#endif
        for (int i = 0; i < len; i++)
            val[i] =
                val[i] -
                T((UINT_TYPE(1) << (BITLENGTH - fractional_bits - 1)));  // substract 2^l-1 to reverse previous addition

    delete[] r_mk2;
    delete[] r_msb;
    delete[] c_prime;
    delete[] b;
}

template <typename T>
static void trunc_pr_in_place(T* val, const int len)
{
    for (int i = 0; i < len; i++)
    {
        val[i] *= UINT_TYPE(1);
        /* val[i].prepare_trunc_share(); // Worth to try out */
    }
    T::communicate();
    for (int i = 0; i < len; i++)
    {
        val[i].complete_public_mult_fixed();
    }
}
