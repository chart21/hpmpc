#pragma once
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../protocols/Protocols.h"
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

#if FUNCTION_IDENTIFIER == 24
#define FUNCTION \
    Naive_Tiled_Intersection_Bench  // Naive intersection of two sets of secret numbers, one set is assumed to be tiled.
                                    // n NUM_INPUTS = 1 intersection of tiled input a and non-tiled input b.
// Bitsliced Function -> Intersection size is TILZE_SIZE*DATTTYPE. Can be combined with split-roles and multiprocessing
// to efficiently compute intersection of large sets with small tiles. Tile size can be adjusted. Assumes that secret
// shares of inputs are already available.
#elif FUNCTION_IDENTIFIER == \
    25  // Bitsliced AES (Reference Code: USUBA). n NUM_INPUTS = DATTYPE*n AES encryptions of blocksize 128.
#include "AES_BS.hpp"
#define FUNCTION AES_Bench
#elif FUNCTION_IDENTIFIER == 26 || FUNCTION_IDENTIFIER == 27 || FUNCTION_IDENTIFIER == 28
#include "../functions/log_reg.hpp"
// Important: Using vectorization, split-roles and multiprocessing will train independent models. Setting DATTYPE =
// BITLENGTH, Threads = 1 and not using Split-roles will train a single model without any optimizations.
#define FUNCTION \
    Logistic_Regression_Bench  // Logistic Regression, n NUM_INPUTS = n samples, DATTYPE/BITLENGTH independent models,
                               // number of features and training iterations can be adjusted.
#elif FUNCTION_IDENTIFIER == 29 || FUNCTION_IDENTIFIER == 30 || FUNCTION_IDENTIFIER == 31
#define FUNCTION \
    Private_Auction_Bench  // Private Auction, n NUM_INPUTS = n bids/offers, DATTYPE/BITLENGTH independent auctions,
                           // price_range is the number of possible distinct prices, can be adjusted.
// Important: Using vectorization, split-roles and multiprocessing will conduct multiple independent auctions. Setting
// DATTYPE = BITLENGTH, Threads = 1 and not using Split-roles will train a single model without any optimizations.
#elif FUNCTION_IDENTIFIER == 32
#include "../functions/intersect_bool.hpp"
#define FUNCTION \
    Naive_Intersection_Bench  // Naive intersection of two sets of secret numbers. n NUM_INPUTS = 1 intersection of
                              // input a and input b, both having length n. Setting DATTYPE = BITLENGTH, Threads = 1 and
                              // not using Split-roles will perform a single intersection without the tiling
                              // optimization from function 57.
#endif

#if DATTYPE > 1
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/k_sint.hpp"
#endif

// Boilerplate
#define RESULTTYPE DATATYPE

#if FUNCTION_IDENTIFIER == 24
template <typename Share>
void Naive_Tiled_Intersection_Bench(DATATYPE* res)
{
    Share::communicate();  // dummy round
    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    const int tile = 100;
    assert(tile <= NUM_INPUTS);
    auto a = new Bitset[tile];  // ideally, a is a smaller subarray of a larger array, then tile intersects can be
                                // computed in parallel
    auto b = new Bitset[NUM_INPUTS];
    auto result = new Bitset[tile];
    intersect(a, b, result, tile, NUM_INPUTS);

    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 25
template <typename Share>
void AES_Bench(DATATYPE* res)
{
    Share::communicate();  // dummy round
    using S = XOR_Share<DATATYPE, Share>;
    auto plain = new S[128][NUM_INPUTS];
    auto key = new S[11][128][NUM_INPUTS];
    auto cipher = new S[128][NUM_INPUTS];
    AES__<S>(plain, key, cipher);
    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 26 || FUNCTION_IDENTIFIER == 27 || FUNCTION_IDENTIFIER == 28
template <typename Share>
void Logistic_Regression_Bench(DATATYPE* res)
{
    Share::communicate();  // dummy round
    auto X_Shared = new Additive_Share<DATATYPE, Share>[NUM_INPUTS][NUM_FEATURES];
    auto y_Shared = new Additive_Share<DATATYPE, Share>[NUM_INPUTS];
    auto weights = new Additive_Share<DATATYPE, Share>[NUM_FEATURES];
    logistic_regression<Share>(X_Shared, y_Shared, weights);
    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 29 || FUNCTION_IDENTIFIER == 30 || FUNCTION_IDENTIFIER == 31
template <typename Share>
void Private_Auction_Bench(DATATYPE* res)
{
    Share::communicate();  // dummy round
    const int price_range = 100;
    using A = Additive_Share<DATATYPE, Share>;
    using S = XOR_Share<DATATYPE, Share>;
    auto offers = new A[NUM_INPUTS][price_range];
    auto bids = new A[NUM_INPUTS][price_range];
    auto accum = new A[price_range * 2]{0};
    for (int i = 0; i < price_range * 2; i += 2)
    {
        for (int j = 0; j < NUM_INPUTS; j++)
        {
            accum[i] += offers[j][i / 2];
            accum[i + 1] += bids[j][i / 2];
        }
    }
    auto clearing_prices = new A[price_range];
    // compute pairwise min of supply and demand for each price
    max_min_sint<0, BITLENGTH>(accum, 2, clearing_prices, price_range, false);
    // compute max of all possible clearing prices
    A result;
    /* argmax_argmin_sint<0, BITLENGTH>(clearing_prices, price_range, &result, 1, true); */
    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 32
template <typename Share>
void Naive_Intersection_Bench(DATATYPE* res)
{
    Share::communicate();  // dummy round
    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    auto a = new Bitset[NUM_INPUTS];
    auto b = new Bitset[NUM_INPUTS];
    auto result = new Bitset[NUM_INPUTS];
    intersect_bool(a, b, result, NUM_INPUTS, NUM_INPUTS);

    dummy_reveal<Share>();
}
#endif
