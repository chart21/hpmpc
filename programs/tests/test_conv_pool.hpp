#pragma once
// Unit tests for the convolution and pooling layers, exercising the exact code
// paths used by the NN inference (Conv2d / AvgPool2d / MaxPool2d / BatchNorm2d).
// Inputs are SECRET-SHARED (weights from P_0, data from P_1) so the shares carry
// real random lambdas — this reproduces the random-mask truncation behaviour of a
// real network (a previous public-value version masked nothing and was misleading).
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/GEMM.hpp"
#include "../functions/Relu.hpp"
#include "../functions/max_min.hpp"
#include "../functions/prob_div.hpp"
#include "headers/config.h"
#include "headers/simple_nn.h"
#include "test_helper.hpp"

#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION test_conv_pool
#endif

#define TEST_CONV 1
#define TEST_AVGPOOL 1
#define TEST_MAXPOOL 1
#define TEST_BATCHNORM 1
#define TEST_RELU_AVG 1   // ReLU->AvgPool, exercises the FUSE_RELU_AVG optimization path
#define TEST_RELU_LARGE 1 // ReLU on LARGER-magnitude values (>1) — exercises the MSB carry chain
                          // through the integer bits, which the tiny (<1) inputs above don't.

using namespace simple_nn;
using namespace Eigen;

// Secret-share n fixed-point values (held by party P) into dst[0..n).
template <int P, typename A>
void share_vals(A* dst, const UINT_TYPE* vals, int n)
{
    for (int i = 0; i < n; i++)
        dst[i].template prepare_receive_from<P>(PROMOTE(vals[i]));
    A::communicate();
    for (int i = 0; i < n; i++)
        dst[i].template complete_receive_from<P>();
    A::communicate();
}

// Re-mask via x*1 (integer, no truncation) so each value gets a *split* lambda, like a real layer
// output (post-ReLU). A raw input share leaves lambda 0 on one party, which is not representative.
template <typename A>
void remask(A* dst, int n)
{
    A one = A(1);
    for (int i = 0; i < n; i++) dst[i] = dst[i].prepare_mult(one);
    A::communicate();
    for (int i = 0; i < n; i++) dst[i].complete_mult_without_trunc();
    A::communicate();
}

#if TEST_CONV == 1
template <typename Share>
bool conv_test()
{
    using A = Additive_Share<DATATYPE, Share>;
    using FFC = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>;
    const int vectorization_factor = DATTYPE / BITLENGTH;

    A::communicate();

    const int batch = 1, ic = 1, oc = 1, ih = 4, iw = 4, ks = 3, stride = 1, pad = 0;
    const int oh = (ih - ks) / stride + 1;  // 2
    const int ow = (iw - ks) / stride + 1;  // 2
    const int osize = batch * oc * oh * ow;

    // small image-like inputs in [0,1]; small (mixed-sign) weights -> small +/- outputs
    float in_f[ih * iw];
    for (int i = 0; i < ih * iw; i++) in_f[i] = (i + 1) / 20.0f;  // 0.05 .. 0.80
    float ker_f[ks * ks] = {0.1, -0.1, 0.1, -0.1, 0.2, -0.1, 0.1, -0.1, 0.1};

    Conv2d<A> conv(ic, oc, ks, stride, pad, /*use_bias=*/false);
    conv.set_layer({batch, ic, ih, iw});
    MatX<A> input(batch * ic, ih * iw);

    UINT_TYPE ker_v[oc * ic * ks * ks];
    for (int i = 0; i < oc * ic * ks * ks; i++) ker_v[i] = FFC::float_to_ufixed(ker_f[i]);
    UINT_TYPE in_v[ih * iw];
    for (int i = 0; i < ih * iw; i++) in_v[i] = FFC::float_to_ufixed(in_f[i]);

    share_vals<P_0>(conv.kernel.data(), ker_v, oc * ic * ks * ks);  // weights from model owner
    share_vals<P_1>(input.data(), in_v, ih * iw);                   // data from data owner
    remask(input.data(), ih * iw);  // activation behaves like a layer output (split lambda)

    conv.forward(input, false);

    alignas(sizeof(DATATYPE)) UINT_TYPE output[osize][vectorization_factor];
    reveal_and_store(conv.output.data(), output, osize);

    float expected[osize];
    for (int oi = 0; oi < oh; oi++)
        for (int oj = 0; oj < ow; oj++)
        {
            float s = 0;
            for (int ki = 0; ki < ks; ki++)
                for (int kj = 0; kj < ks; kj++)
                {
                    int ii = oi * stride + ki - pad;
                    int jj = oj * stride + kj - pad;
                    if (ii >= 0 && ii < ih && jj >= 0 && jj < iw)
                        s += in_f[ii * iw + jj] * ker_f[ki * ks + kj];
                }
            expected[oi * ow + oj] = s;
        }

    bool ok = true;
    for (int q = 0; q < osize; q++)
        for (int v = 0; v < vectorization_factor; v++)
        {
            float got = FFC::ufixed_to_float(output[q][v]);
            print_compare(expected[q], got, epsilon);
            if (got - expected[q] > epsilon || got - expected[q] < -epsilon)
                ok = false;
        }
    return ok;
}
#endif

#if TEST_AVGPOOL == 1
template <typename Share>
bool avgpool_test()
{
    using A = Additive_Share<DATATYPE, Share>;
    using FFC = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>;
    const int vectorization_factor = DATTYPE / BITLENGTH;

    A::communicate();

    const int batch = 1, ch = 1, ih = 4, iw = 4, ks = 2, stride = 2, pad = 0;
    const int oh = (ih - ks) / stride + 1;  // 2
    const int ow = (iw - ks) / stride + 1;  // 2
    const int osize = batch * ch * oh * ow;

    // small positive inputs in [0,1] (avg pool follows ReLU, so inputs are >= 0)
    float in_f[ih * iw];
    for (int i = 0; i < ih * iw; i++) in_f[i] = (i + 1) / 20.0f;  // 0.05 .. 0.80

    AvgPool2d<A> pool(ks, stride, pad);
    pool.set_layer({batch, ch, ih, iw});

    MatX<A> input(batch * ch, ih * iw);
    UINT_TYPE in_v[ih * iw];
    for (int i = 0; i < ih * iw; i++) in_v[i] = FFC::float_to_ufixed(in_f[i]);
    share_vals<P_1>(input.data(), in_v, ih * iw);
    remask(input.data(), ih * iw);  // avg pool input behaves like a layer output (split lambda)

    pool.forward(input, false);

    alignas(sizeof(DATATYPE)) UINT_TYPE output[osize][vectorization_factor];
    reveal_and_store(pool.output.data(), output, osize);

    float expected[osize];
    for (int oi = 0; oi < oh; oi++)
        for (int oj = 0; oj < ow; oj++)
        {
            float s = 0;
            for (int y = 0; y < ks; y++)
                for (int x = 0; x < ks; x++)
                {
                    int ii = oi * stride + y - pad;
                    int jj = oj * stride + x - pad;
                    if (ii >= 0 && ii < ih && jj >= 0 && jj < iw)
                        s += in_f[ii * iw + jj];
                }
#if FUSE_RELU_AVG == 1
            expected[oi * ow + oj] = s;  // a standalone avgpool skips its division when fused (a preceding ReLU does it)
#else
            expected[oi * ow + oj] = s / (ks * ks);
#endif
        }

    bool ok = true;
    for (int q = 0; q < osize; q++)
        for (int v = 0; v < vectorization_factor; v++)
        {
            float got = FFC::ufixed_to_float(output[q][v]);
            print_compare(expected[q], got, epsilon);
            if (got - expected[q] > epsilon || got - expected[q] < -epsilon)
                ok = false;
        }
    return ok;
}
#endif

#if TEST_MAXPOOL == 1
template <typename Share>
bool maxpool_test()
{
    using A = Additive_Share<DATATYPE, Share>;
    using FFC = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>;
    const int vectorization_factor = DATTYPE / BITLENGTH;

    A::communicate();

    const int batch = 1, ch = 1, ih = 4, iw = 4, ks = 2, stride = 2, pad = 0;
    const int oh = (ih - ks) / stride + 1;  // 2
    const int ow = (iw - ks) / stride + 1;  // 2
    const int osize = batch * ch * oh * ow;

    // small positive inputs in [0,1]
    float in_f[ih * iw] = {0.10, 0.20, 0.30, 0.40, 0.80, 0.60, 0.70, 0.50,
                           0.15, 0.90, 0.50, 0.70, 0.25, 0.45, 0.65, 0.35};

    MaxPool2d<A> pool(ks, stride, pad);
    pool.set_layer({batch, ch, ih, iw});

    MatX<A> input(batch * ch, ih * iw);
    UINT_TYPE in_v[ih * iw];
    for (int i = 0; i < ih * iw; i++) in_v[i] = FFC::float_to_ufixed(in_f[i]);
    share_vals<P_1>(input.data(), in_v, ih * iw);
    remask(input.data(), ih * iw);

    pool.forward(input, false);

    alignas(sizeof(DATATYPE)) UINT_TYPE output[osize][vectorization_factor];
    reveal_and_store(pool.output.data(), output, osize);

    float expected[osize];
    for (int oi = 0; oi < oh; oi++)
        for (int oj = 0; oj < ow; oj++)
        {
            float mx = -1e9;
            for (int y = 0; y < ks; y++)
                for (int x = 0; x < ks; x++)
                {
                    int ii = oi * stride + y - pad;
                    int jj = oj * stride + x - pad;
                    if (ii >= 0 && ii < ih && jj >= 0 && jj < iw)
                        if (in_f[ii * iw + jj] > mx)
                            mx = in_f[ii * iw + jj];
                }
            expected[oi * ow + oj] = mx;
        }

    bool ok = true;
    for (int q = 0; q < osize; q++)
        for (int v = 0; v < vectorization_factor; v++)
        {
            float got = FFC::ufixed_to_float(output[q][v]);
            print_compare(expected[q], got, epsilon);
            if (got - expected[q] > epsilon || got - expected[q] < -epsilon)
                ok = false;
        }
    return ok;
}
#endif

#if TEST_BATCHNORM == 1
template <typename Share>
bool batchnorm_test()
{
    using A = Additive_Share<DATATYPE, Share>;
    using FFC = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>;
    const int vectorization_factor = DATTYPE / BITLENGTH;

    A::communicate();

    // Inference BatchNorm: output = move_var * (x - move_mu) + beta  (per channel).
    const int batch = 1, ch = 2, h = 2, w = 2, hw = h * w;
    const int osize = batch * ch * hw;

    // small inputs in [0,1] (BN typically follows a conv whose outputs are small)
    float in_f[ch * hw] = {0.2, 0.3, 0.4, 0.5, 0.4, 0.6, 0.8, 0.9};  // [c=0 ; c=1]
    float mu_f[ch] = {0.3, 0.5};
    float scale_f[ch] = {0.5, 0.25};  // move_var = gamma / sqrt(var + eps)
    float beta_f[ch] = {0.1, 0.2};

    BatchNorm2d<A> bn;
    bn.set_layer({batch, ch, h, w});

    UINT_TYPE mu_v[ch], scale_v[ch], beta_v[ch], gamma_v[ch];
    for (int c = 0; c < ch; c++)
    {
        mu_v[c] = FFC::float_to_ufixed(mu_f[c]);
        scale_v[c] = FFC::float_to_ufixed(scale_f[c]);
        beta_v[c] = FFC::float_to_ufixed(beta_f[c]);
        gamma_v[c] = FFC::float_to_ufixed(1.0);
    }
    share_vals<P_0>(bn.move_mu.data(), mu_v, ch);
    share_vals<P_0>(bn.move_var.data(), scale_v, ch);
    share_vals<P_0>(bn.beta.data(), beta_v, ch);
    share_vals<P_0>(bn.gamma.data(), gamma_v, ch);

    MatX<A> input(batch * ch, hw);
    UINT_TYPE in_v[ch * hw];
    for (int i = 0; i < ch * hw; i++) in_v[i] = FFC::float_to_ufixed(in_f[i]);
    share_vals<P_1>(input.data(), in_v, ch * hw);
    remask(input.data(), ch * hw);

    bn.forward(input, false);

    alignas(sizeof(DATATYPE)) UINT_TYPE output[osize][vectorization_factor];
    reveal_and_store(bn.output.data(), output, osize);

    float expected[osize];
    for (int c = 0; c < ch; c++)
        for (int j = 0; j < hw; j++)
            expected[c * hw + j] = scale_f[c] * (in_f[c * hw + j] - mu_f[c]) + beta_f[c];

    bool ok = true;
    for (int q = 0; q < osize; q++)
        for (int v = 0; v < vectorization_factor; v++)
        {
            float got = FFC::ufixed_to_float(output[q][v]);
            print_compare(expected[q], got, epsilon);
            if (got - expected[q] > epsilon || got - expected[q] < -epsilon)
                ok = false;
        }
    return ok;
}
#endif

#if TEST_RELU_AVG == 1
// ReLU followed by AvgPool. With FUSE_RELU_AVG==1, SimpleNN::compile() sets the avgpool denominator on
// the ReLU; the ReLU then folds the 1/denominator division into its share-conversion truncation and the
// AvgPool skips its own division. This test mimics that wiring so we can validate the optimization on a
// tiny input instead of a full network. Plaintext reference is avg(relu(x)).
template <typename Share>
bool relu_avgpool_test()
{
    using A = Additive_Share<DATATYPE, Share>;
    using FFC = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>;
    const int vectorization_factor = DATTYPE / BITLENGTH;

    A::communicate();

    const int batch = 1, ch = 1, ih = 4, iw = 4, ks = 2, stride = 2, pad = 0;
    const int oh = (ih - ks) / stride + 1;  // 2
    const int ow = (iw - ks) / stride + 1;  // 2
    const int osize = batch * ch * oh * ow;

    // mixed-sign inputs so ReLU actually zeroes some values
    float in_f[ih * iw] = {-0.30, 0.20, 0.50, -0.10, 0.40, -0.60, 0.80, 0.30,
                           -0.20, 0.90, -0.50, 0.70, 0.10, 0.45, -0.65, 0.35};

    ReLU<A> relu;
    relu.set_layer({batch, ch, ih, iw});
    AvgPool2d<A> pool(ks, stride, pad);
    pool.set_layer({batch, ch, ih, iw});
#if FUSE_RELU_AVG == 1
    relu.set_fused_avgpool_denominator(pool.average_denominator());  // same wiring as SimpleNN::compile()
#endif

    MatX<A> input(batch * ch, ih * iw);
    UINT_TYPE in_v[ih * iw];
    for (int i = 0; i < ih * iw; i++) in_v[i] = FFC::float_to_ufixed(in_f[i]);
    share_vals<P_1>(input.data(), in_v, ih * iw);
    remask(input.data(), ih * iw);

    // call forward via base pointers: with FUSE_RELU_AVG==1 ReLU::forward is (quirkily) private,
    // but the network dispatches through Layer*, so do the same here.
    Layer<A>* relu_ptr = &relu;
    Layer<A>* pool_ptr = &pool;
    relu_ptr->forward(input, false);
    pool_ptr->forward(relu.output, false);

    alignas(sizeof(DATATYPE)) UINT_TYPE output[osize][vectorization_factor];
    reveal_and_store(pool.output.data(), output, osize);

    float expected[osize];
    for (int oi = 0; oi < oh; oi++)
        for (int oj = 0; oj < ow; oj++)
        {
            float s = 0;
            for (int y = 0; y < ks; y++)
                for (int x = 0; x < ks; x++)
                {
                    int ii = oi * stride + y - pad;
                    int jj = oj * stride + x - pad;
                    if (ii >= 0 && ii < ih && jj >= 0 && jj < iw)
                    {
                        float v = in_f[ii * iw + jj];
                        s += (v > 0 ? v : 0);
                    }
                }
            expected[oi * ow + oj] = s / (ks * ks);
        }

    bool ok = true;
    for (int q = 0; q < osize; q++)
        for (int v = 0; v < vectorization_factor; v++)
        {
            float got = FFC::ufixed_to_float(output[q][v]);
            print_compare(expected[q], got, epsilon);
            if (got - expected[q] > epsilon || got - expected[q] < -epsilon)
                ok = false;
        }
    return ok;
}
#endif

#if TEST_RELU_LARGE == 1
// ReLU on values with magnitude > 1, so the sign/MSB computation runs the full carry chain through the
// integer bits. The tiny (<1) inputs in the other tests leave those bits 0, hiding MSB-adder bugs (e.g.
// the RESHARE_OPT reshared adders). Plaintext reference is max(x, 0).
template <typename Share>
bool relu_large_test()
{
    using A = Additive_Share<DATATYPE, Share>;
    using FFC = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>;
    const int vectorization_factor = DATTYPE / BITLENGTH;

    A::communicate();

    const int n = 16;
    float in_f[n] = {-7.5, 6.2, -3.1, 0.9, -0.4, 4.7, -6.8, 2.3,
                     5.5, -5.5, 1.1, -1.1, 7.0, -7.0, 0.2, -0.2};

    ReLU<A> relu;
    relu.set_layer({1, 1, 4, 4});  // 16 values
    Layer<A>* relu_ptr = &relu;

    MatX<A> input(1, n);
    UINT_TYPE in_v[n];
    for (int i = 0; i < n; i++) in_v[i] = FFC::float_to_ufixed(in_f[i]);
    share_vals<P_1>(input.data(), in_v, n);
    remask(input.data(), n);

    relu_ptr->forward(input, false);

    alignas(sizeof(DATATYPE)) UINT_TYPE output[n][vectorization_factor];
    reveal_and_store(relu.output.data(), output, n);

    bool ok = true;
    for (int q = 0; q < n; q++)
        for (int v = 0; v < vectorization_factor; v++)
        {
            float expected = in_f[q] > 0 ? in_f[q] : 0;
            float got = FFC::ufixed_to_float(output[q][v]);
            print_compare(expected, got, epsilon);
            if (got - expected > epsilon || got - expected < -epsilon)
                ok = false;
        }
    return ok;
}
#endif

template <typename Share>
bool test_conv_pool(DATATYPE* res)
{
    int num_tests = 0;
    int num_passed = 0;

#if TEST_CONV == 1
    test_function(num_tests, num_passed, "Convolution", conv_test<Share>);
#endif
#if TEST_AVGPOOL == 1
    test_function(num_tests, num_passed, "AvgPool", avgpool_test<Share>);
#endif
#if TEST_MAXPOOL == 1
    test_function(num_tests, num_passed, "MaxPool", maxpool_test<Share>);
#endif
#if TEST_BATCHNORM == 1
    test_function(num_tests, num_passed, "BatchNorm", batchnorm_test<Share>);
#endif
#if TEST_RELU_AVG == 1
    test_function(num_tests, num_passed, "ReLU+AvgPool(fused)", relu_avgpool_test<Share>);
#endif
#if TEST_RELU_LARGE == 1
    test_function(num_tests, num_passed, "ReLU(large)", relu_large_test<Share>);
#endif

    print_stats(num_tests, num_passed);
    if (num_tests == num_passed)
        return true;
    return false;
}
