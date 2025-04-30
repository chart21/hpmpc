#pragma once
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "bench_helper.hpp"
// Each circuit will be evaluated in parallel, specified by NUM_PROCESSES. And additionally the Split-Roles mulitplier
// and vectorization multiplier. Split-roles multipliers: 1 (3-PC): 6 2 (3-PC -> 4-PC) 24, 3 (4-PC): 24

// Vectorization multipliers depend on the functions and are just state explicitly in the comments of the function
// definitions. Vectorization multipliers (Example for BITLENGTH = 32): DATTYPE = 32: 1 DATTYPE = 128: 4 DATTYPE = 256:
// 8 DATTYPE = 512: 16

// For instance, evaluating arithmetic operations such as MULT bench with NUM_INPUTS=100, 3-PC split-roles,
// NUM_PROCESSES=4,DATTYPE=256,BITLENGTH=32 will evaluate 100*6*4*8 = 19200 AND gates in parallel. Evaluating Boolean
// oeprations such as AND Bench with the same parameters will evaluate 100*6*4*256 = 614400 AND gates in parallel.

#if FUNCTION_IDENTIFIER == 1
#define FUNCTION AND_BENCH  // AND gates benchmark, n NUM_INPUTS = n*(DATTYPE/BITLENGTH) AND gates
#elif FUNCTION_IDENTIFIER == 2
#define FUNCTION MULT_BENCH  // Integer multiplication, n NUM_INPUTS = n*(DATTYPE/BITLENGTH) multiplications
#elif FUNCTION_IDENTIFIER == 3
#define FUNCTION MULT_BENCH  // Fixed point multiplication, n NUM_INPUTS = n*(DATTYPE/BITLENGTH) multiplications
#elif FUNCTION_IDENTIFIER == 4
#define FUNCTION \
    DIV_BENCH  // Fixed point division using Newton-Raphson approximation, n NUM_INPUTS = n*(DATTYPE/BITLENGTH)
               // divisions. Number of iterations for Newton-Raphson approximation can be adjusted.
#elif FUNCTION_IDENTIFIER == 5
#define FUNCTION SHARE_BENCH  // Secret Sharing of inputs, n NUM_INPUTS = n*(DATTYPE/BITLENGTH) Inputs shared by P_0
#elif FUNCTION_IDENTIFIER == 6
#define FUNCTION REVEAL_BENCH  // Reveal of inputs, n NUM_INPUTS = n*(DATTYPE/BITLENGTH) Inputs revealed to all parties
#elif FUNCTION_IDENTIFIER == 7
#define FUNCTION dot_prod_bench  // Matrix Vector Product n NUM_INPUTS = n*(DATTYPE/BITLENGTH) product output size
#endif

#if BITLENGTH > 1
#include "../../datatypes/Additive_Share.hpp"
#endif

// Boilerplate
#define RESULTTYPE DATATYPE

#if FUNCTION_IDENTIFIER == 1
template <typename Share>
void AND_BENCH(DATATYPE* res)
{
    using S = XOR_Share<DATATYPE, Share>;
    auto a = new S[NUM_INPUTS];
    auto b = new S[NUM_INPUTS];
    auto c = new S[NUM_INPUTS];
    Share::communicate();  // dummy round
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        c[i] = a[i] & b[i];
    }
    Share::communicate();

    for (int i = 0; i < NUM_INPUTS; i++)
    {
        c[i].complete_and();
    }

    Share::communicate();

    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 2 || FUNCTION_IDENTIFIER == 3
template <typename Share>
void MULT_BENCH(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    auto a = new S[NUM_INPUTS];
    auto b = new S[NUM_INPUTS];
    auto c = new S[NUM_INPUTS];
    Share::communicate();  // dummy round
    for (int i = 0; i < NUM_INPUTS; i++)
    {
#if FUNCTION_IDENTIFIER == 2  // int
        c[i] = a[i].prepare_mult(b[i]);
#elif FUNCTION_IDENTIFIER == 3  // fixed
        c[i] = a[i].prepare_dot(b[i]);
        c[i].mask_and_send_dot();
#endif
    }
    Share::communicate();
    for (int i = 0; i < NUM_INPUTS; i++)
    {
#if FUNCTION_IDENTIFIER == 2  // int
        c[i].complete_mult_without_trunc();
#elif FUNCTION_IDENTIFIER == 3  // fixed
        c[i].complete_mult();
#endif
    }
    Share::communicate();

    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 4
template <typename Share>
void DIV_BENCH(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    auto a = new S[NUM_INPUTS];
    auto b = new S[NUM_INPUTS];
    auto c = new S[NUM_INPUTS];
    const int n = 8;       // iterations for Newton-Raphson division
    Share::communicate();  // dummy round

    // y0(x) = 3e^(0.5−x) + 0.003 -> initial guess
    for (int i = 0; i < NUM_INPUTS; i++)
        c[i] = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(3 * exp(0.5) + 0.003);
    // Newpthon Raphson formula 1/x = limn→∞ yn = y_n−1(2 − xyn−1)

    for (int j = 0; j < n; j++)
    {
        for (int i = 0; i < NUM_INPUTS; i++)
        {
            c[i] = c[i] + c[i] - b[i].prepare_dot(c[i]);
            c[i].mask_and_send_dot();
        }
        Share::communicate();
        for (int i = 0; i < NUM_INPUTS; i++)
            c[i].complete_mult();
    }
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        c[i] = a[i].prepare_dot(c[i]);
        c[i].mask_and_send_dot();
    }
    Share::communicate();
    for (int i = 0; i < NUM_INPUTS; i++)
        c[i].complete_mult();

    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 5
template <typename Share>
void SHARE_BENCH(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    auto inputs = new A[NUM_INPUTS];
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        inputs[i].template prepare_receive_from<P_0>(SET_ALL_ZERO());
    }
    Share::communicate();
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        inputs[i].template complete_receive_from<P_0>();
    }

    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 6
template <typename Share>
void REVEAL_BENCH(DATATYPE* res)
{
    Share::communicate();  // dummy round
    using A = Additive_Share<DATATYPE, Share>;
    auto inputs = new A[NUM_INPUTS];
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        inputs[i].prepare_reveal_to_all();
    }
    Share::communicate();
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        inputs[i].complete_reveal_to_all();
    }

    dummy_reveal<Share>();
}
#endif

template <typename Share>
void dot_prod_bench(DATATYPE* res)
{
    Share::communicate();  // dummy round
    using A = Additive_Share<DATATYPE, Share>;
    auto a = new A[NUM_INPUTS];
    auto b = new A[NUM_INPUTS][NUM_INPUTS];
    auto c = new A[NUM_INPUTS];
    Share::communicate();  // dummy round
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        for (int j = 0; j < NUM_INPUTS; j++)
        {
            c[i] += a[i].prepare_dot(b[i][j]);
        }
        c[i].mask_and_send_dot_without_trunc();
    }
    Share::communicate();
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        c[i].complete_mult_without_trunc();
    }

    Share::communicate();
    dummy_reveal<Share>();

    delete[] a;
    delete[] b;
    delete[] c;
}
