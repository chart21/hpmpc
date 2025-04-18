#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/GEMM.hpp"
#include "bench_helper.hpp"
#include "headers/config.h"
#include "headers/simple_nn.h"

#if FUNCTION_IDENTIFIER == 33 || FUNCTION_IDENTIFIER == 34 || FUNCTION_IDENTIFIER == 35
#define FUNCTION conv_2D_bench
#elif FUNCTION_IDENTIFIER == 36
#define FUNCTION mat_mul_bench
#elif FUNCTION_IDENTIFIER == 37 || FUNCTION_IDENTIFIER == 38 || FUNCTION_IDENTIFIER == 39
#include "../functions/Relu.hpp"
#define FUNCTION ReLU_bench
#elif FUNCTION_IDENTIFIER == 40
#define FUNCTION avg_pool_bench
#elif FUNCTION_IDENTIFIER == 41
#define FUNCTION batch_norm_bench
#elif FUNCTION_IDENTIFIER == 42 || FUNCTION_IDENTIFIER == 43 || FUNCTION_IDENTIFIER == 44
#include "../functions/adders/ppa_msb_4_way.hpp"
#include "../functions/adders/ppa_msb_unsafe.hpp"
#include "../functions/adders/rca_msb.hpp"
#define FUNCTION boolean_adder_bench
#elif FUNCTION_IDENTIFIER == 45 || FUNCTION_IDENTIFIER == 46 || FUNCTION_IDENTIFIER == 47
#define FUNCTION conv_2D_bench
#endif

#define RESULTTYPE DATATYPE
using namespace std;
using namespace simple_nn;
using namespace Eigen;

const int num_repeat = 20;

void generateElements() {}

#if FUNCTION_IDENTIFIER == 33 || FUNCTION_IDENTIFIER == 34 || FUNCTION_IDENTIFIER == 35
template <typename Share>
void conv_2D_bench(DATATYPE* res)
{
    for (int i = 0; i < num_repeat; ++i)
    {
        using S = Additive_Share<DATATYPE, Share>;
        Share::communicate();  // dummy round
#if FUNCTION_IDENTIFIER == 33
                               // CryptGPU figure 1 a: Conv2D layer: nxnx3 input, 11x11 kernel, 64 output channels,
                               // stride 4, padding 2
        using A = Additive_Share<DATATYPE, Share>;
        const int batch = 1;
        Conv2d<S> conv(3, 64, 11, 4, 2);
        vector<int> input_shape = {batch, 3, NUM_INPUTS, NUM_INPUTS};
        MatX<S> input(batch, 3 * NUM_INPUTS * NUM_INPUTS);
#elif FUNCTION_IDENTIFIER == 34
                               // CryptGPU figure 1 b: Conv2D layer: 32x32x3 input, 11x11 kernel, 64 output channels,
                               // stride 4, padding 2, batch size k
        const int batch = NUM_INPUTS;
        Conv2d<S> conv(3, 64, 11, 4, 2);
        vector<int> input_shape = {batch, 3, 32, 32};
        MatX<S> input(batch, 3 * 32 * 32);
#elif FUNCTION_IDENTIFIER == 35
                               // CryptGPU figure 1 c: Conv2D layer: nxnx512 input, 3x3 kernel, 512 output channels,
                               // stride 1, padding 1, batch size k
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

#if FUNCTION_IDENTIFIER == 36
// Piranha figure 4: NxN matrix multiplication
template <typename Share>
void mat_mul_bench(DATATYPE* res)
{
    for (int i = 0; i < num_repeat; ++i)
    {
        using S = Additive_Share<DATATYPE, Share>;
        const int batch = 1;
        Share::communicate();  // dummy round
        MatX<S> OM(NUM_INPUTS, NUM_INPUTS);
        MatX<S> AM(NUM_INPUTS, NUM_INPUTS);
        MatX<S> BM(NUM_INPUTS, NUM_INPUTS);
#if USE_CUDA_GEMM == 0  // Use CPU GEMM
        const int TILE_SIZE = 64;
        for (int i = 0; i < BM.size(); ++i)
            OM(i) = S(0);
        auto A = AM.data();
        MatX<S> BMT = BM.transpose();
        auto B = BMT.data();
        auto C = OM.data();
        const int m = NUM_INPUTS;
        const int p = NUM_INPUTS;
        const int f = NUM_INPUTS;
        for (int i = 0; i < m; i += TILE_SIZE)
        {
            int i_max = std::min(i + TILE_SIZE, m);
            for (int j = 0; j < p; j += TILE_SIZE)
            {
                int j_max = std::min(j + TILE_SIZE, p);
                for (int k = 0; k < f; k += TILE_SIZE)
                {
                    int k_max = std::min(k + TILE_SIZE, f);
                    for (int ii = i; ii < i_max; ++ii)
                    {
                        const int iip = ii * p;
                        const int iif = ii * f;
                        for (int jj = j; jj < j_max; ++jj)
                        {
                            const int jjf = jj * f;
                            auto temp = S(0);
                            for (int kk = k; kk < k_max; ++kk)
                            {
#if PUBLIC_WEIGHTS == 0
                                temp += A[iif + kk].prepare_dot(B[jjf + kk]);
#else
                                temp += A[iif + kk].mult_public(B[jjf + kk]);
#endif
                            }
                            C[iip + jj] += temp;
                        }
                    }
                }
                for (int ii = i; ii < i_max; ++ii)
                {
                    const int row = ii * p;
                    for (int jj = j; jj < j_max; ++jj)
                    {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
                        C[row + jj].mask_and_send_dot_without_trunc();
#else
                        C[row + jj].mask_and_send_dot();
#endif
#else
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
#else
                        C[row + jj] = C[row + jj].prepare_mult_public_fixed(1);  // initiate truncation
#endif
#endif
                    }
                }
            }
        }

#elif USE_CUDA_GEMM > 0  // Use CUDA GEMM
        auto A = AM.data();
        auto B = BM.data();
        auto C = OM.data();
        int mul_m = NUM_INPUTS;
        int mul_n = NUM_INPUTS;
        int mul_k = NUM_INPUTS;

        T::GEMM(A, B, C, mul_m, mul_n, mul_k, true);
        for (int j = 0; j < NUM_INPUTS * NUM_INPUTS; ++j)
        {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
            C[j].mask_and_send_dot_without_trunc();
#else
            C[j].mask_and_send_dot();
#endif
#else
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
#else
            C[j] = C[j].prepare_mult_public_fixed(1);  // initiate truncation
#endif
#endif
        }
#endif
        S::communicate();
#if USE_CUDA_GEMM == 0
        for (int i = 0; i < m; i += TILE_SIZE)
        {
            int i_max = std::min(i + TILE_SIZE, m);
            for (int j = 0; j < p; j += TILE_SIZE)
            {
                int j_max = std::min(j + TILE_SIZE, p);
                for (int ii = i; ii < i_max; ++ii)
                {
                    const int row = ii * p;
                    for (int jj = j; jj < j_max; ++jj)
                    {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
                        C[row + jj].complete_mult_without_trunc();
#else
                        C[row + jj].complete_mult();
#endif
#else
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
#else
                        C[row + jj].complete_mult_public_fixed();
#endif
#endif
                    }
                }
            }
        }
#elif USE_CUDA_GEMM > 0
        for (int i = 0; i < NUM_INPUTS * NUM_INPUTS; ++i)
        {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
            C[i].complete_mult_without_trunc();
#else
            C[i].complete_mult();
#endif
#else
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH == 1
#else
            C[i].complete_mult_public_fixed();
#endif
#endif
        }

#endif
        dummy_reveal<Share>();
    }
}
#endif

#if FUNCTION_IDENTIFIER == 37 || FUNCTION_IDENTIFIER == 38 || FUNCTION_IDENTIFIER == 39
template <typename Share>
void ReLU_bench(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    auto prev_out = new S[NUM_INPUTS];
    auto out = new S[NUM_INPUTS];
    const int m = REDUCED_BITLENGTH_m;
    const int k = REDUCED_BITLENGTH_k;
    Share::communicate();  // dummy round
    RELU<m, k>(prev_out, prev_out + NUM_INPUTS, out);
    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 40
template <typename Share>
void avg_pool_bench(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    Share::communicate();  // dummy round
    auto layer = new AvgPool2d<S>(2, 2);
    const int batch = 1;
    vector<int> input_shape = {batch, 3, NUM_INPUTS, NUM_INPUTS};
    MatX<S> input(batch, 3 * NUM_INPUTS * NUM_INPUTS);
    layer->set_layer(input_shape);
    layer->forward(input, false);
    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 41
template <typename Share>
void batch_norm_bench(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    Share::communicate();  // dummy round
    const int batch = 1;
    const int num_channels = 64;
    auto batch_norm = new BatchNorm2d<S>();
    vector<int> input_shape = {batch, num_channels, NUM_INPUTS, NUM_INPUTS};
    MatX<S> input(batch * num_channels, NUM_INPUTS * NUM_INPUTS);
    batch_norm->set_layer(input_shape);
    batch_norm->forward(input, false);
    dummy_reveal<Share>();
}
#endif

#if FUNCTION_IDENTIFIER == 42 || FUNCTION_IDENTIFIER == 43 || FUNCTION_IDENTIFIER == 44
template <typename Share>
void boolean_adder_bench(DATATYPE* res)
{
    const int m = REDUCED_BITLENGTH_m;
    const int k = REDUCED_BITLENGTH_k;
    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<k - m, S>;

    Share::communicate();  // dummy round
    S* y = new S[NUM_INPUTS];
    Bitset* s1 = new Bitset[NUM_INPUTS];
    Bitset* s2 = new Bitset[NUM_INPUTS];
#if BANDWIDTH_OPTIMIZED == 1 && ONLINE_OPTIMIZED == 0
    std::vector<BooleanAdder_MSB<k - m, S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 1
    std::vector<PPA_MSB_4Way<k - m, S>> adders;
#elif BANDWIDTH_OPTIMIZED == 0 && ONLINE_OPTIMIZED == 0
    /* std::vector<PPA_MSB<k-m,S>> adders; */
    std::vector<PPA_MSB_Unsafe<k - m, S>> adders;
#endif
    /* std::vector<PPA_MSB_4Way<k,S>> adders; */
    adders.reserve(NUM_INPUTS);
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], y[i]);
    }

    while (!adders[0].is_done())
    {
        for (int i = 0; i < NUM_INPUTS; i++)
        {
            adders[i].step();
        }
        Share::communicate();
    }
    delete[] s1;
    delete[] s2;
    adders.clear();
    adders.shrink_to_fit();
}
#endif

#if FUNCTION_IDENTIFIER == 45 || FUNCTION_IDENTIFIER == 46 || FUNCTION_IDENTIFIER == 47

template <typename Share>
void conv_2D_bench(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    Share::communicate();  // dummy round
    const int batch = 1;
#if FUNCTION_IDENTIFIER == 45
    auto conv = new Conv2d<S>(3, 64, 11, 4, 2);
#elif FUNCTION_IDENTIFIER == 46
    auto conv = new Conv2d<S>(3, 64, 3, 1, 1);
#elif FUNCTION_IDENTIFIER == 47
    auto conv = new Conv2d<S>(64, 64, 3, 1, 1);
#endif
    vector<int> input_shape = {batch, 64, NUM_INPUTS, NUM_INPUTS};
    MatX<S> input(batch, 64 * NUM_INPUTS * NUM_INPUTS);
    conv->set_layer(input_shape);
    conv->forward(input, false);
    dummy_reveal<Share>();
}
#endif
