#pragma once
#include <cstdint>
#include <cstring>
#include <iostream>
#include <bitset>
#include <cmath>
#include <sys/types.h>
#include "../../protocols/Protocols.h"
#include "../../protocols/XOR_Share.hpp"
#include "../../protocols/Additive_Share.hpp"
#include "../../protocols/Matrix_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include "boolean_adder_msb.hpp"
#include "ppa_msb.hpp"
#include "ppa_msb_unsafe.hpp"
#include "ppa_msb_4_way.hpp"
#include "../../utils/print.hpp"
#include "headers/simple_nn.h"
#include "headers/config.h"
#include "Relu.hpp"

#if FUNCTION_IDENTIFIER == 400 || FUNCTION_IDENTIFIER == 401 || FUNCTION_IDENTIFIER == 402
#define FUNCTION conv_2D_bench
#elif FUNCTION_IDENTIFIER == 403
#define FUNCTION mat_mul_bench
#elif FUNCTION_IDENTIFIER == 404 || FUNCTION_IDENTIFIER == 405 || FUNCTION_IDENTIFIER == 406
#define FUNCTION ReLU_bench
#elif FUNCTION_IDENTIFIER == 407 || FUNCTION_IDENTIFIER == 408 || FUNCTION_IDENTIFIER == 409
#define FUNCTION ReLU_sint_bench
#endif

#if FUNCTION_IDENTIFIER == 404 || FUNCTION_IDENTIFIER == 407 //RCA
#define BANDWIDTH_OPTIMIZED 1
#define ONLINE_OPTIMIZED 0
#elif FUNCTION_IDENTIFIER == 405 || FUNCTION_IDENTIFIER == 408 //PPA
#define BANDWIDTH_OPTIMIZED 0
#define ONLINE_OPTIMIZED 0
#elif FUNCTION_IDENTIFIER == 406 || FUNCTION_IDENTIFIER == 409 //PPA 4-way
#define BANDWIDTH_OPTIMIZED 0
#define ONLINE_OPTIMIZED 1
#endif

#define RESULTTYPE DATATYPE
using namespace std;
using namespace simple_nn;
using namespace Eigen;

const int num_repeat = 20;

void generateElements()
{

}

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

#if FUNCTION_IDENTIFIER == 400 || FUNCTION_IDENTIFIER == 401 || FUNCTION_IDENTIFIER == 402
template<typename Share>
void conv_2D_bench(DATATYPE* res)
{
for(int i = 0; i < num_repeat; ++i)
{
using S = Additive_Share<DATATYPE, Share>;
Share::communicate(); // dummy round
#if FUNCTION_IDENTIFIER == 400
    // CryptGPU figure 1 a: Conv2D layer: nxnx3 input, 11x11 kernel, 64 output channels, stride 4, padding 2
using A = Additive_Share<DATATYPE, Share>;
const int batch = 1;
Conv2d<S> conv(3, 64, 11, 4, 2);
vector<int> input_shape = {batch, 3, NUM_INPUTS, NUM_INPUTS};
MatX<S> input(batch, 3 * NUM_INPUTS * NUM_INPUTS);
#elif FUNCTION_IDENTIFIER == 401
    // CryptGPU figure 1 b: Conv2D layer: 32x32x3 input, 11x11 kernel, 64 output channels, stride 4, padding 2, batch size k
const int batch = NUM_INPUTS;
Conv2d<S> conv(3, 64, 11, 4, 2);
vector<int> input_shape = {batch, 3, 32, 32};
MatX<S> input(batch, 3 * 32 * 32);
#elif FUNCTION_IDENTIFIER == 402
    // CryptGPU figure 1 c: Conv2D layer: nxnx512 input, 3x3 kernel, 512 output channels, stride 1, padding 1, batch size k
const int batch = 1;
Conv2d<S> conv(512, 512, 3, 1, 1);
vector<int> input_shape = {batch, 512, NUM_INPUTS, NUM_INPUTS};
MatX<S> input(batch, 512 * NUM_INPUTS * NUM_INPUTS);
#endif
conv.set_layer(input_shape);
conv.forward(input, false);
dummy_reveal<Share>();

}
}
#endif

#if FUNCTION_IDENTIFIER == 403
// Piranha figure 4: NxN matrix multiplication
template<typename Share>
void mat_mul_bench(DATATYPE* res)
{
    for(int i = 0; i < num_repeat; ++i)
    {
    using S = Additive_Share<DATATYPE, Share>;
    const int batch = 1; 
    Share::communicate(); // dummy round
    MatX<S> OM(NUM_INPUTS, NUM_INPUTS);
    MatX<S> AM(NUM_INPUTS, NUM_INPUTS);
    MatX<S> BM(NUM_INPUTS, NUM_INPUTS);
    const int TILE_SIZE = 64;
    for(int i = 0; i < BM.size(); ++i)
    OM(i) = S(0);
    auto A = AM.data();
    MatX<S> BMT = BM.transpose();
    auto B = BMT.data();
    auto C = OM.data();
    const int m = NUM_INPUTS;
    const int p = NUM_INPUTS;
    const int f = NUM_INPUTS;
    for (int i = 0; i < m; i += TILE_SIZE) {
            int i_max = std::min(i + TILE_SIZE, m);
            for (int j = 0; j < p; j += TILE_SIZE) {
                int j_max = std::min(j + TILE_SIZE, p);
                for (int k = 0; k < f; k += TILE_SIZE) {
                    int k_max = std::min(k + TILE_SIZE, f);
                    for (int ii = i; ii < i_max; ++ii) {
                        const int iip = ii*p;
                        const int iif = ii*f;
                        for (int jj = j; jj < j_max; ++jj) {
                            const int jjf = jj*f;
                            auto temp = S(0);
                            for (int kk = k; kk < k_max; ++kk) {
                                #if PUBLIC_WEIGHTS == 0
                                    temp += A[iif+kk].prepare_dot(B[jjf + kk]);
                                #else
                                    temp += A[iif+kk].mult_public(B[jjf + kk]);
                                #endif
                            }
                            C[iip + jj] += temp;
                        }
                    }
                }
                for (int ii = i; ii < i_max; ++ii) {
                    const int row = ii*p;
                    for (int jj = j; jj < j_max; ++jj) {
                    #if PUBLIC_WEIGHTS == 0
                    #if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
                            C[row + jj].mask_and_send_dot_without_trunc();
                    #else
                            C[row + jj].mask_and_send_dot();
                    #endif
                    #else
                        #if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
                        #else
                            C[row + jj].prepare_mult_public_fixed(1); //initiate truncation
                        #endif
                    #endif
            }
        }
    }
    }
    S::communicate();
    for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
                for (int ii = i; ii < i_max; ++ii) {
                    const int row = ii*p;
                    for (int jj = j; jj < j_max; ++jj) {
                    #if PUBLIC_WEIGHTS == 0
                    #if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
                                    C[row+jj].complete_mult_without_trunc();
                    #else
                                    C[row+jj].complete_mult();
                    #endif
                    #else
                        #if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
                        #else
                                    C[row+jj].complete_mult_public_fixed();
                        #endif
                    #endif
                    }
                }
        }
    }
    dummy_reveal<Share>();
    }
}
#endif

#if FUNCTION_IDENTIFIER == 404 || FUNCTION_IDENTIFIER == 405 || FUNCTION_IDENTIFIER == 406  
template<typename Share>
void ReLU_bench(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    auto prev_out = new S[NUM_INPUTS];
    auto out = new S[NUM_INPUTS];
    const int m = REDUCED_BITLENGTH_m;
    const int k = REDUCED_BITLENGTH_k;
    Share::communicate(); // dummy round
    RELU<m,k>(prev_out, prev_out + NUM_INPUTS, out);
    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 407 || FUNCTION_IDENTIFIER == 408 || FUNCTION_IDENTIFIER == 409
// sint uses massive batch size
template<typename Share>
void ReLU_sint_bench(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;
    auto prev_out = new sint[NUM_INPUTS];
    auto out = new sint[NUM_INPUTS];
    const int m = REDUCED_BITLENGTH_m;
    const int k = REDUCED_BITLENGTH_k;
    Share::communicate(); // dummy round
    RELU<m,k>(prev_out, prev_out + NUM_INPUTS, out);
    dummy_reveal<Share>();
}
#endif
