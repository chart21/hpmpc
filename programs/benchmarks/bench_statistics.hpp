#pragma once
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/max_min.hpp"
#include "../functions/share_conversion.hpp"
#include "bench_helper.hpp"
// Each circuit will be evaluated in parallel, specified by NUM_PROCESSES. And additionally the Split-Roles mulitplier
// and vectorization multiplier. Split-roles multipliers: 1 (3-PC): 6 2 (3-PC -> 4-PC) 24, 3 (4-PC): 24

// Vectorization multipliers depend on the functions and are just state explicitly in the comments of the function
// definitions. Vectorization multipliers (Example for BITLENGTH = 32): DATTYPE = 32: 1 DATTYPE = 128: 4 DATTYPE = 256:
// 8 DATTYPE = 512: 16

// For instance, evaluating arithmetic operations such as MULT bench with NUM_INPUTS=100, 3-PC split-roles,
// NUM_PROCESSES=4,DATTYPE=256,BITLENGTH=32 will evaluate 100*6*4*8 = 19200 AND gates in parallel. Evaluating Boolean
// oeprations such as AND Bench with the same parameters will evaluate 100*6*4*256 = 614400 AND gates in parallel.

#if FUNCTION_IDENTIFIER == 13 || FUNCTION_IDENTIFIER == 14 || FUNCTION_IDENTIFIER == 15
#define FUNCTION \
    COMP_BENCH  // Batched comparison of two secret numbers, n NUM_INPUTS = n*DATTYPE comparisons (!). Functions use
                // RCA, PPA, or PPA 4-Way respectively
#elif FUNCTION_IDENTIFIER == 16 || FUNCTION_IDENTIFIER == 17 || FUNCTION_IDENTIFIER == 18
#define FUNCTION \
    MAXMIN_BENCH  // Maximum of a range of secret numbers, n NUM_INPUTS = DATTYPE/BITLENGTH maximums of n inputs.
                  // Functions use RCA, PPA, or PPA 4-Way respectively
#elif FUNCTION_IDENTIFIER == 19 || FUNCTION_IDENTIFIER == 20 || FUNCTION_IDENTIFIER == 21
#define FUNCTION \
    MAXMIN_BENCH  // Minimum of a range of secret numbers, n NUM_INPUTS = DATTYPE/BITLENGTH minimums of n inputs.
                  // Functions use RCA, PPA, or PPA 4-Way respectively
#elif FUNCTION_IDENTIFIER == 22
#define FUNCTION \
    AVG_BENCH  // Fixed point average of a range of secret numbers, n NUM_INPUTS = DATTYPE/BITLENGTH average of n inputs
#elif FUNCTION_IDENTIFIER == 23
#define FUNCTION SUM_BENCH  // Sum of a range of secret numbers, n NUM_INPUTS = DATTYPE/BITLENGTH sums of n inputs
#endif

#if DATTYPE > 1
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/k_sint.hpp"
#include "../functions/intersect_bool.hpp"
#endif

// Boilerplate
#define RESULTTYPE DATATYPE

#if FUNCTION_IDENTIFIER == 13 || FUNCTION_IDENTIFIER == 14 || FUNCTION_IDENTIFIER == 15
template <typename Share>
void COMP_BENCH(DATATYPE* res)
{
    // c = (a > b) = msb(b-a)
    Share::communicate();  // dummy round
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;  // Share conversion is currently only supported in minimal batch sizes of size DATTYPE
    auto a = new sint[NUM_INPUTS];
    auto b = new sint[NUM_INPUTS];
    auto tmp = new sint[NUM_INPUTS];
    auto c = new S[NUM_INPUTS];
    const int k = BITLENGTH;  // Reducing k will make the calculation probabilistic
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        tmp[i] = b[i] - a[i];
    }
    get_msb_range<0, k>(tmp, c, NUM_INPUTS);
    dummy_reveal<Share>();
}

#elif FUNCTION_IDENTIFIER == 16 || FUNCTION_IDENTIFIER == 17 || FUNCTION_IDENTIFIER == 18 || \
    FUNCTION_IDENTIFIER == 19 || FUNCTION_IDENTIFIER == 20 || FUNCTION_IDENTIFIER == 21
template <typename Share>
void MAXMIN_BENCH(DATATYPE* res)
{
    Share::communicate();  // dummy round
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;
    const int k = BITLENGTH;  // Reducing k will make the calculation probabilistic
    auto inputs = new A[NUM_INPUTS];
    A result;
#if FUNCTION_IDENTIFIER == 16 || FUNCTION_IDENTIFIER == 17 || FUNCTION_IDENTIFIER == 18
    max_min_sint<0, k>(inputs, NUM_INPUTS, &result, 1, true);
#elif FUNCTION_IDENTIFIER == 19 || FUNCTION_IDENTIFIER == 20 || FUNCTION_IDENTIFIER == 21
    max_min_sint<0, k>(inputs, NUM_INPUTS, &result, 1, false);
#endif
    delete[] inputs;

    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 22
template <typename Share>
void AVG_BENCH(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    auto inputs = new A[NUM_INPUTS];
    A c = 0;
    Share::communicate();  // dummy round
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        c += inputs[i];
    }
    c = c.prepare_mult_public_fixed(
        FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1.f / (float)NUM_INPUTS));
    Share::communicate();
    c.complete_public_mult_fixed();

    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 23
template <typename Share>
void SUM_BENCH(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    auto inputs = new A[NUM_INPUTS];
    A c = 0;
    Share::communicate();  // dummy round
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        c += inputs[i];
    }

    dummy_reveal<Share>();
}
#endif
