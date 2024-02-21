#pragma once
#include <cstdint>
#include <cstring>
#include <iostream>
#include <cmath>
#include "../../protocols/Protocols.h"
#include "../../protocols/XOR_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../utils/print.hpp"
//Each circuit will be evaluated in parallel, specified by NUM_PROCESSES. And additionally the Split-Roles mulitplier and vectorization multiplier.
//Split-roles multipliers: 
//1 (3-PC): 6
//2 (3-PC -> 4-PC) 24,
//3 (4-PC): 24 


//Vectorization multipliers depend on the functions and are just state explicitly in the comments of the function definitions.
//Vectorization multipliers (Example for BITLENGTH = 32):
//DATTYPE = 32: 1
//DATTYPE = 128: 4
//DATTYPE = 256: 8
//DATTYPE = 512: 16

// For instance, evaluating arithmetic operations such as MULT bench with NUM_INPUTS=100, 3-PC split-roles, NUM_PROCESSES=4,DATTYPE=256,BITLENGTH=32 will evaluate 100*6*4*8 = 19200 AND gates in parallel.
// Evaluating Boolean oeprations such as AND Bench with the same parameters will evaluate 100*6*4*256 = 614400 AND gates in parallel.

#if FUNCTION_IDENTIFIER == 40
#define FUNCTION AND_BENCH //AND gates benchmark, n NUM_INPUTS = n*(DATTYPE/BITLENGTH) AND gates
#elif FUNCTION_IDENTIFIER == 41
#define FUNCTION MULT_BENCH //Integer multiplication, n NUM_INPUTS = n*(DATTYPE/BITLENGTH) multiplications
#elif FUNCTION_IDENTIFIER == 42
#define FUNCTION MULT_BENCH //Fixed point multiplication, n NUM_INPUTS = n*(DATTYPE/BITLENGTH) multiplications
#elif FUNCTION_IDENTIFIER == 43
#define FUNCTION DIV_BENCH //Fixed point division using Newton-Raphson approximation, n NUM_INPUTS = n*(DATTYPE/BITLENGTH) divisions. Number of iterations for Newton-Raphson approximation can be adjusted.
#elif FUNCTION_IDENTIFIER == 44
#define FUNCTION SHARE_BENCH // Secret Sharing of inputs, n NUM_INPUTS = n*(DATTYPE/BITLENGTH) Inputs shared by P_0
#elif FUNCTION_IDENTIFIER == 45
#define FUNCTION REVEAL_BENCH // Reveal of inputs, n NUM_INPUTS = n*(DATTYPE/BITLENGTH) Inputs revealed to all parties
#elif FUNCTION_IDENTIFIER == 46 || FUNCTION_IDENTIFIER == 47 || FUNCTION_IDENTIFIER == 48
#define FUNCTION COMP_BENCH // Batched comparison of two secret numbers, n NUM_INPUTS = n*DATTYPE comparisons (!). Functions use RCA, PPA, or PPA 4-Way respectively
#elif FUNCTION_IDENTIFIER == 49 || FUNCTION_IDENTIFIER == 50 || FUNCTION_IDENTIFIER == 51
#define FUNCTION MAXMIN_BENCH // Maximum of a range of secret numbers, n NUM_INPUTS = DATTYPE/BITLENGTH maximums of n inputs. Functions use RCA, PPA, or PPA 4-Way respectively
#elif FUNCTION_IDENTIFIER == 52 || FUNCTION_IDENTIFIER == 53 || FUNCTION_IDENTIFIER == 54 
#define FUNCTION MAXMIN_BENCH  // Minimum of a range of secret numbers, n NUM_INPUTS = DATTYPE/BITLENGTH minimums of n inputs. Functions use RCA, PPA, or PPA 4-Way respectively
#elif FUNCTION_IDENTIFIER == 55
#define FUNCTION AVG_BENCH // Fixed point average of a range of secret numbers, n NUM_INPUTS = DATTYPE/BITLENGTH average of n inputs
#elif FUNCTION_IDENTIFIER == 56
#define FUNCTION SUM_BENCH // Sum of a range of secret numbers, n NUM_INPUTS = DATTYPE/BITLENGTH sums of n inputs
#elif FUNCTION_IDENTIFIER == 57
#define FUNCTION Naive_Tiled_Intersection_Bench // Naive intersection of two sets of secret numbers, one set is assumed to be tiled. n NUM_INPUTS = 1 intersection of tiled input a and non-tiled input b.
//Bitsliced Function -> Intersection size is TILZE_SIZE*DATTTYPE. Can be combined with split-roles and multiprocessing to efficiently compute intersection of large sets with small tiles. Tile size can be adjusted. Assumes that secret shares of inputs are already available.
#elif FUNCTION_IDENTIFIER == 58 // Bitsliced AES (Reference Code: USUBA). n NUM_INPUTS = DATTYPE*n AES encryptions of blocksize 128.
#include "AES_BS_SHORT.hpp"
#define FUNCTION AES_Bench
#elif FUNCTION_IDENTIFIER == 59 || FUNCTION_IDENTIFIER == 60 || FUNCTION_IDENTIFIER == 61
#include "log_reg.hpp" 
//Important: Using vectorization, split-roles and multiprocessing will train independent models. Setting DATTYPE = BITLENGTH, Threads = 1 and not using Split-roles will train a single model without any optimizations.
#define FUNCTION Logistic_Regression_Bench // Logistic Regression, n NUM_INPUTS = n samples, DATTYPE/BITLENGTH independent models, number of features and training iterations can be adjusted.  
#elif FUNCTION_IDENTIFIER == 62 || FUNCTION_IDENTIFIER == 63 || FUNCTION_IDENTIFIER == 64
#define FUNCTION Private_Auction_Bench // Private Auction, n NUM_INPUTS = n bids/offers, DATTYPE/BITLENGTH independent auctions, price_range is the number of possible distinct prices, can be adjusted.
//Important: Using vectorization, split-roles and multiprocessing will conduct multiple independent auctions. Setting DATTYPE = BITLENGTH, Threads = 1 and not using Split-roles will train a single model without any optimizations.
#elif FUNCTION_IDENTIFIER == 65
#include "intersect_bool.hpp"
#define FUNCTION Naive_Intersection_Bench // Naive intersection of two sets of secret numbers. n NUM_INPUTS = 1 intersection of input a and input b, both having length n. Setting DATTYPE = BITLENGTH, Threads = 1 and not using Split-roles will perform a single intersection without the tiling optimization from function 57.
#endif


#if DATTYPE > 1
#include "sevare_helper.hpp"
#include "../../protocols/Additive_Share.hpp"
#include "../../datatypes/k_sint.hpp"
#endif


//Boilerplate
#define RESULTTYPE DATATYPE
void generateElements()
{}

//if placed after a function, gurantees that all parties have finished computation and communication
template<typename Share>
void dummy_reveal()
{
    using S = XOR_Share<DATATYPE, Share>;
    S dummy;
    dummy.prepare_reveal_to_all();
    Share::communicate();
    dummy.complete_reveal_to_all();
}

#if FUNCTION_IDENTIFIER == 40
template<typename Share>
void AND_BENCH(DATATYPE* res)
{
    using S = XOR_Share<DATATYPE, Share>;
    auto a = new S[NUM_INPUTS];
    auto b = new S[NUM_INPUTS];
    auto c = new S[NUM_INPUTS];
    Share::communicate(); // dummy round
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        c[i] = a[i] & b[i];
    }
    Share::communicate();

    for(int i = 0; i < NUM_INPUTS; i++)
    {
        c[i].complete_and();
    }

    Share::communicate();

    dummy_reveal<Share>();

}
#endif

#if FUNCTION_IDENTIFIER == 41 || FUNCTION_IDENTIFIER == 42
template<typename Share>
void MULT_BENCH(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    auto a = new S[NUM_INPUTS];
    auto b = new S[NUM_INPUTS];
    auto c = new S[NUM_INPUTS];
    Share::communicate(); // dummy round
    for(int i = 0; i < NUM_INPUTS; i++)
    {
    #if FUNCTION_IDENTIFIER == 40 // int
        c[i] = a[i] * b[i];
    #elif FUNCTION_IDENTIFIER == 41 // fixed
        c[i] = a[i].prepare_dot(b[i]);
        c[i].mask_and_send_dot();
    #endif

    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
    #if FUNCTION_IDENTIFIER == 40 // int
        c[i].complete_mult_without_trunc();
    #elif FUNCTION_IDENTIFIER == 41 // fixed
        c[i].complete_mult();
    #endif
    }
    Share::communicate();

    dummy_reveal<Share>();
}
#endif


#if FUNCTION_IDENTIFIER == 43
template<typename Share>
void DIV_BENCH(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    auto a = new S[NUM_INPUTS];
    auto b = new S[NUM_INPUTS];
    auto c = new S[NUM_INPUTS];
    const int n = 8; //iterations for Newton-Raphson division
    Share::communicate(); // dummy round

    //y0(x) = 3e^(0.5−x) + 0.003 -> initial guess
    for(int i = 0; i < NUM_INPUTS; i++)
        c[i] = 3*exp(0.5) + 0.003; //TODO: convert to fixed point, value is valid for a large input domain of b
    // Newpthon Raphson formula 1/x = limn→∞ yn = y_n−1(2 − xyn−1)
   
    for(int j = 0; j < n; j++)
        {
        for(int i = 0; i < NUM_INPUTS; i++)
        {
            c[i] = c[i] + c[i] - b[i].prepare_dot(c[i]);
            c[i].mask_and_send_dot();
        }
        Share::communicate();
        for(int i = 0; i < NUM_INPUTS; i++)
            c[i].complete_mult();
        }
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        c[i] = a[i].prepare_dot(c[i]);
        c[i].mask_and_send_dot();
    } 
        Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
        c[i].complete_mult();

    dummy_reveal<Share>();
}
#endif


#if FUNCTION_IDENTIFIER == 44
template<typename Share>
void SHARE_BENCH(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    auto inputs = new A[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        inputs[i].template prepare_receive_from<P_0>(SET_ALL_ZERO());
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        inputs[i].template complete_receive_from<P_0>();
    }

    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 45
template<typename Share>
void REVEAL_BENCH(DATATYPE* res)
{
    Share::communicate(); // dummy round
    using A = Additive_Share<DATATYPE, Share>;
    auto inputs = new A[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        inputs[i].prepare_reveal_to_all();
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        inputs[i].complete_reveal_to_all();
    }

    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 46 || FUNCTION_IDENTIFIER == 47 || FUNCTION_IDENTIFIER == 48
template<typename Share>
void COMP_BENCH(DATATYPE* res)
{
    // c = (a > b) = msb(b-a)
    Share::communicate(); // dummy round
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>; //Share conversion is currently only supported in minimal batch sizes of size DATTYPE
    auto a = new sint[NUM_INPUTS];
    auto b = new sint[NUM_INPUTS];
    auto tmp = new sint[NUM_INPUTS];
    auto c = new S[NUM_INPUTS];
    const int k = BITLENGTH; //Reducing k will make the calculation probabilistic
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        tmp[i] = b[i] - a[i];
    }
    get_msb_range<0, k>(tmp, c, NUM_INPUTS);
    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 49 || FUNCTION_IDENTIFIER == 50 || FUNCTION_IDENTIFIER == 51 || FUNCTION_IDENTIFIER == 52 || FUNCTION_IDENTIFIER == 53 || FUNCTION_IDENTIFIER == 54
template<typename Share>
void MAXMIN_BENCH(DATATYPE *res)
{
Share::communicate(); // dummy round
using S = XOR_Share<DATATYPE, Share>;
using A = Additive_Share<DATATYPE, Share>;
using sint = sint_t<A>;
const int k = BITLENGTH; //Reducing k will make the calculation probabilistic
auto inputs = new A[NUM_INPUTS];
A result;
#if FUNCTION_IDENTIFIER == 49 || FUNCTION_IDENTIFIER == 50 || FUNCTION_IDENTIFIER == 51
max_min_sint<0,k>(inputs, NUM_INPUTS, &result, 1, true);
#elif FUNCTION_IDENTIFIER == 52 || FUNCTION_IDENTIFIER == 53 || FUNCTION_IDENTIFIER == 54
max_min_sint<0,k>(inputs, NUM_INPUTS, &result, 1, false);
#endif
delete[] inputs;

dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 55
template<typename Share>
void AVG_BENCH(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    auto inputs = new A[NUM_INPUTS];
    A c = 0;
    Share::communicate(); // dummy round
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        c+=inputs[i];
    }
    c *= UINT_TYPE(1.0/NUM_INPUTS); //TODO: convert 1/NUM_INPUTS to fixed point
    Share::communicate();
    c.complete_public_mult_fixed();

    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 56
template<typename Share>
void SUM_BENCH(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    auto inputs = new A[NUM_INPUTS];
    A c = 0;
    Share::communicate(); // dummy round
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        c+=inputs[i];
    }

    dummy_reveal<Share>();
}
#endif



#if FUNCTION_IDENTIFIER == 57
template<typename Share>
void Naive_Tiled_Intersection_Bench(DATATYPE *res)
{
   Share::communicate(); // dummy round
   using S = XOR_Share<DATATYPE, Share>; 
   using Bitset = sbitset_t<BITLENGTH,S>;
   const int tile = 100;
   assert(tile <= NUM_INPUTS);
   auto a = new Bitset[tile]; // ideally, a is a smaller subarray of a larger array, then tile intersects can be computed in parallel
   auto b = new Bitset[NUM_INPUTS];
   auto result = new Bitset[tile];
   intersect(a, b, result, tile, NUM_INPUTS);

   dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 58
template<typename Share>
void AES_Bench(DATATYPE *res)
{
    Share::communicate(); // dummy round
    using S = XOR_Share<DATATYPE, Share>;
    auto plain = new S[128][NUM_INPUTS]; 
    auto key = new S[11][128][NUM_INPUTS];
    auto cipher = new S[128][NUM_INPUTS];
    AES__<S>(plain, key, cipher);
    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 59 || FUNCTION_IDENTIFIER == 60 || FUNCTION_IDENTIFIER == 61
template<typename Share>
void Logistic_Regression_Bench(DATATYPE *res)
{
    Share::communicate(); // dummy round
    auto X_Shared = new Additive_Share<DATATYPE, Share>[NUM_INPUTS][NUM_FEATURES];
    auto y_Shared = new Additive_Share<DATATYPE, Share>[NUM_INPUTS];
    auto weights = new Additive_Share<DATATYPE, Share>[NUM_FEATURES];
    logistic_regression<Share>(X_Shared, y_Shared, weights);
    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 62 || FUNCTION_IDENTIFIER == 63 || FUNCTION_IDENTIFIER == 64
template<typename Share>
void Private_Auction_Bench(DATATYPE *res)
{
    Share::communicate(); // dummy round
    const int price_range = 100;
    using A = Additive_Share<DATATYPE, Share>;
    using S = XOR_Share<DATATYPE, Share>;
    auto offers = new A[NUM_INPUTS][price_range];
    auto bids = new A[NUM_INPUTS][price_range];
    auto accum = new A[price_range*2]{0};
    for (int i = 0; i < price_range*2; i+=2) {
        for (int j = 0; j < NUM_INPUTS; j++) {
            accum[i] += offers[j][i/2];
            accum[i+1] += bids[j][i/2];
        }
    }
    auto clearing_prices = new A[price_range];
    //compute pairwise min of supply and demand for each price
    max_min_sint<0, BITLENGTH>(accum, 2, clearing_prices, price_range, false);
    //compute max of all possible clearing prices
    A result;
    /* argmax_argmin_sint<0, BITLENGTH>(clearing_prices, price_range, &result, 1, true); */
    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 65
template<typename Share>
void Naive_Intersection_Bench(DATATYPE *res)
{
   Share::communicate(); // dummy round
   using S = XOR_Share<DATATYPE, Share>; 
   using Bitset = sbitset_t<BITLENGTH,S>;
   auto a = new Bitset[NUM_INPUTS]; 
   auto b = new Bitset[NUM_INPUTS];
   auto result = new Bitset[NUM_INPUTS];
   intersect_bool(a, b, result, NUM_INPUTS, NUM_INPUTS);

   dummy_reveal<Share>();
}
#endif
