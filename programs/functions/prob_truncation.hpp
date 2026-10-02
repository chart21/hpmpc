#pragma once
#include "../../datatypes/Additive_Share.hpp"
template <typename T>
static void trunc_pr_in_place(T* val, const int len);

template <typename T>
static void trunc_2k_in_place(T* val, const int len, bool isPositive = false, int fractional_bits = FRACTIONAL)
{
#if TS1_FUSED_ACTIVE
    // 2PC: TS1 is fused into the ReLUs (RELU_range_in_place_ts1), a stand-alone TS1 would need a message of its own.
    // TS_Mix (TRUNC_APPROACH 4) truncates everything else probabilistically.
    (void) isPositive;
#if TRUNC_APPROACH == 1
    (void) val, (void) len, (void) fractional_bits;
    fprintf(stderr, "TRUNC_APPROACH 1 (2PC): a truncation outside a ReLU, use TRUNC_APPROACH 4\n");
    std::abort();
#else
    if (fractional_bits != FRACTIONAL)
    {
        fprintf(stderr, "TRUNC_APPROACH 4 (2PC): a truncation by %d bits outside a ReLU\n", fractional_bits);
        std::abort();
    }
    trunc_pr_in_place(val, len);
#endif
#else

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
#endif
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

#if PROTOCOL == 4 && PUBLIC_WEIGHTS == 1 && DATAOWNER != -1
// trunc_pr_in_place for the output of the FIRST layer under PUBLIC_WEIGHTS (g_a_known_input): its input was the raw
// data-owner share (the non-owner's mask is 0) and the public weights multiply locally, so the value still sits
// entirely in the data owner's share, where the SecureML local truncation wraps for positive values (see GEMM.hpp).
// The owner truncates it in the clear with an arithmetic shift and re-masks; same single message as trunc_pr.
template <typename T>
static void trunc_a_known_in_place(T* val, const int len)
{
    for (int i = 0; i < len; i++)
        val[i] = val[i].prepare_mult_public_fixed_a_known(1);
    T::communicate();
    for (int i = 0; i < len; i++)
        val[i].complete_mult_public_fixed_a_known();
}
#endif
