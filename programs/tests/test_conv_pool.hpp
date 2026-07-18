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


// Model parameters: secretly shared from the model owner, or plainly assigned under PUBLIC_WEIGHTS.
template <int P, typename W>
void set_weights(W* dst, const UINT_TYPE* vals, int n)
{
#if PUBLIC_WEIGHTS == 1
    for (int i = 0; i < n; i++) dst[i] = vals[i];
#else
    share_vals<P>(dst, vals, n);
#endif
}

// Re-mask via x*1 (integer, no truncation) so each value gets a *split* lambda, like a real layer
// output (post-ReLU). A raw input share leaves lambda 0 on one party, which is not representative.
// Uses the BAKED mask/send (linear call order) so a directly following MSB-adder batch (ReLU etc.)
// finds the reshare material baked into the masks under RESHARE_OPT_SIM=1 - like a real layer output.
template <typename A>
void remask(A* dst, int n, bool bake = false)
{
#if PROTOCOL == 4 && BEAVER == 1 && PUBLIC_WEIGHTS == 0
    A one = A(1);
    for (int i = 0; i < n; i++)
    {
        dst[i] = dst[i].prepare_dot(one);  // retrieves [lxly] itself; the mask/send below adds only the fresh mask
        dst[i].mask_and_send_dot_baked(bake ? i : -1);
    }
    A::communicate();
    for (int i = 0; i < n; i++) dst[i].complete_mult_without_trunc();
    A::communicate();
#else
    (void)bake;
    A one = A(1);
    for (int i = 0; i < n; i++) dst[i] = dst[i].prepare_mult(one);
    A::communicate();
    for (int i = 0; i < n; i++) dst[i].complete_mult_without_trunc();
    A::communicate();
#endif
}

#if TEST_CONV == 1
template <typename Share>
bool conv_test()
{
    using A = Additive_Share<DATATYPE, Share>;
    using FFC = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>;
    const int vectorization_factor = DATTYPE / BITLENGTH;

    A::communicate();

    // Small but representative: multi-batch + multi-output-channel + padding (ic>1 covered by LeNet conv2).
    const int batch = 2, ic = 1, oc = 3, ih = 6, iw = 6, ks = 3, stride = 1, pad = 1;
    const int oh = (ih + 2 * pad - ks) / stride + 1;
    const int ow = (iw + 2 * pad - ks) / stride + 1;
    const int osize = batch * oc * oh * ow;

    // varied inputs/weights spanning a range of dot magnitudes
    float* in_f = new float[batch * ih * iw];
    for (int i = 0; i < batch * ih * iw; i++) in_f[i] = ((i * 7 + 3) % 40 - 20) / 20.0f;
    float ker_f[oc * ic * ks * ks];
    for (int i = 0; i < oc * ic * ks * ks; i++) ker_f[i] = ((i * 13 + 5) % 14 - 7) / 20.0f;

    Conv2d<A> conv(ic, oc, ks, stride, pad, /*use_bias=*/false);
    conv.set_layer({batch, ic, ih, iw});
    MatX<A> input(batch * ic, ih * iw);

    UINT_TYPE ker_v[oc * ic * ks * ks];
    for (int i = 0; i < oc * ic * ks * ks; i++) ker_v[i] = FFC::float_to_ufixed(ker_f[i]);
    UINT_TYPE* in_v = new UINT_TYPE[batch * ih * iw];
    for (int i = 0; i < batch * ih * iw; i++) in_v[i] = FFC::float_to_ufixed(in_f[i]);

    set_weights<P_0>(conv.kernel.data(), ker_v, oc * ic * ks * ks);  // weights from model owner
    share_vals<P_1>(input.data(), in_v, batch * ih * iw);           // data from data owner
    remask(input.data(), batch * ih * iw);  // activation behaves like a layer output (split lambda)

    conv.forward(input, false);

    auto* output = new UINT_TYPE[osize][DATTYPE / BITLENGTH];
    reveal_and_store(conv.output.data(), output, osize);

    float* expected = new float[osize];
    for (int b = 0; b < batch; b++)
        for (int oc_i = 0; oc_i < oc; oc_i++)
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
                                s += in_f[b * ih * iw + ii * iw + jj] * ker_f[oc_i * ks * ks + ki * ks + kj];
                        }
                    expected[(b * oc + oc_i) * oh * ow + oi * ow + oj] = s;
                }

    int nfail = 0;
    for (int q = 0; q < osize; q++)
        for (int v = 0; v < vectorization_factor; v++)
        {
            float got = FFC::ufixed_to_float(output[q][v]);
            if (got - expected[q] > epsilon || got - expected[q] < -epsilon)
            {
                nfail++;
                if (nfail <= 8)
                    print_online("conv FAIL q=" + std::to_string(q) + " exp=" + std::to_string(expected[q]) +
                                 " got=" + std::to_string(got));
            }
        }
    print_online("conv nfail=" + std::to_string(nfail) + " / " + std::to_string(osize));
    bool ok = (nfail == 0);
    delete[] in_f;
    delete[] in_v;
    delete[] output;
    delete[] expected;
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
    set_weights<P_0>(bn.move_mu.data(), mu_v, ch);
    set_weights<P_0>(bn.move_var.data(), scale_v, ch);
    set_weights<P_0>(bn.beta.data(), beta_v, ch);
    set_weights<P_0>(bn.gamma.data(), gamma_v, ch);

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
    remask(input.data(), ih * iw, /*bake=*/true);  // a (fused) ReLU directly follows

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
    remask(input.data(), n, /*bake=*/true);  // a ReLU (MSB adder batch) directly follows

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

// FC (Linear, with shared bias) -> ReLU: exercises the RESHARE_OPT_SIM=1 baking on the FC path
// (linear-order mask_and_send with sequential triple retrieval) incl. the bias mask compensation.
template <typename Share>
bool fc_relu_test()
{
    using A = Additive_Share<DATATYPE, Share>;
    using FFC = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>;
    const int vf = DATTYPE / BITLENGTH;
    A::communicate();

    const int batch = 2, in_feat = 20, out_feat = 32;  // batch*out_feat = 64 = exact bit-slice groups

    float* in_f = new float[batch * in_feat];
    for (int i = 0; i < batch * in_feat; i++) in_f[i] = ((i * 7 + 3) % 16 - 8) / 8.0f;  // ~ -1..0.875
    float* w_f = new float[out_feat * in_feat];
    for (int i = 0; i < out_feat * in_feat; i++) w_f[i] = ((i * 5 + 1) % 12 - 6) / 12.0f;  // ~ -0.5..0.42
    float* b_f = new float[out_feat];
    for (int i = 0; i < out_feat; i++) b_f[i] = ((i * 3 + 2) % 8 - 4) / 4.0f;  // ~ -1..0.75

    Linear<A> fc(in_feat, out_feat);
    fc.set_layer({batch, in_feat});

    UINT_TYPE* w_v = new UINT_TYPE[out_feat * in_feat];
    for (int i = 0; i < out_feat * in_feat; i++) w_v[i] = FFC::float_to_ufixed(w_f[i]);
    UINT_TYPE* b_v = new UINT_TYPE[out_feat];
    for (int i = 0; i < out_feat; i++) b_v[i] = FFC::float_to_ufixed(b_f[i]);
    UINT_TYPE* in_v = new UINT_TYPE[batch * in_feat];
    for (int i = 0; i < batch * in_feat; i++) in_v[i] = FFC::float_to_ufixed(in_f[i]);

    set_weights<P_0>(fc.W.data(), w_v, out_feat * in_feat);
    set_weights<P_0>(fc.b.data(), b_v, out_feat);
    MatX<A> input(batch, in_feat);
    share_vals<P_1>(input.data(), in_v, batch * in_feat);
    remask(input.data(), batch * in_feat);

    fc.forward(input, false);

    ReLU<A> relu;
    relu.set_layer({batch, out_feat});
    Layer<A>* relu_ptr = &relu;
    relu_ptr->forward(fc.output, false);

    auto* output = new UINT_TYPE[batch * out_feat][DATTYPE / BITLENGTH];
    reveal_and_store(relu.output.data(), output, batch * out_feat);

    int nfail = 0;
    for (int n = 0; n < batch; n++)
        for (int o = 0; o < out_feat; o++)
        {
            float s = b_f[o];
            for (int i = 0; i < in_feat; i++) s += w_f[o * in_feat + i] * in_f[n * in_feat + i];
            float expected = s > 0 ? s : 0;
            for (int v = 0; v < vf; v++)
            {
                float got = FFC::ufixed_to_float(output[n * out_feat + o][v]);
                if (got - expected > epsilon || got - expected < -epsilon)
                {
                    nfail++;
                    if (nfail <= 6)
                        print_online("fc_relu FAIL n=" + std::to_string(n) + " o=" + std::to_string(o) +
                                     " exp=" + std::to_string(expected) + " got=" + std::to_string(got));
                }
            }
        }
    print_online("fc_relu nfail=" + std::to_string(nfail) + " / " + std::to_string(batch * out_feat));
    bool ok = (nfail == 0);
    delete[] in_f; delete[] w_f; delete[] b_f; delete[] w_v; delete[] b_v; delete[] in_v; delete[] output;
    return ok;
}

// Conv -> ReLU: the conv output (masked in mask_and_send_dot) flows through the ReLU MSB adder, so this exercises
// the RESHARE_OPT_SIM=1 baking (rt.a baked into the conv-output mask l, consumed by the reshared MSB adder).
template <typename Share>
bool conv_relu_test()
{
    using A = Additive_Share<DATATYPE, Share>;
    using FFC = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>;
    const int vf = DATTYPE / BITLENGTH;
    A::communicate();

#if RESHARE_OPT == 1 && RESHARE_OPT_SIM == 1 && DATTYPE == BITLENGTH
    // Reset so the summary printed after this test covers ONLY the baked conv->ReLU reshares: the bake must
    // make every one of them match (g_rb_mismatch == 0), which is the exact SIM=0-equivalence condition.
    g_rb_checks = 0; g_rb_mismatch = 0;
#endif
    const int batch = 1, ic = 1, oc = 4, ih = 6, iw = 6, ks = 3, stride = 1, pad = 0;
    const int oh = (ih - ks) / stride + 1, ow = (iw - ks) / stride + 1;
    const int osize = batch * oc * oh * ow;

    float* in_f = new float[batch * ih * iw];
    for (int i = 0; i < batch * ih * iw; i++) in_f[i] = ((i * 5 + 1) % 20 - 10) / 5.0f;  // ~ -2..1.8
    float ker_f[oc * ic * ks * ks];
    for (int i = 0; i < oc * ic * ks * ks; i++) ker_f[i] = ((i * 7 + 2) % 10 - 5) / 5.0f;  // ~ -1..0.8

    Conv2d<A> conv(ic, oc, ks, stride, pad, false);
    conv.set_layer({batch, ic, ih, iw});
    MatX<A> input(batch * ic, ih * iw);
    UINT_TYPE ker_v[oc * ic * ks * ks];
    for (int i = 0; i < oc * ic * ks * ks; i++) ker_v[i] = FFC::float_to_ufixed(ker_f[i]);
    UINT_TYPE* in_v = new UINT_TYPE[batch * ih * iw];
    for (int i = 0; i < batch * ih * iw; i++) in_v[i] = FFC::float_to_ufixed(in_f[i]);
    set_weights<P_0>(conv.kernel.data(), ker_v, oc * ic * ks * ks);
    share_vals<P_1>(input.data(), in_v, batch * ih * iw);
    remask(input.data(), batch * ih * iw);
    conv.forward(input, false);

    ReLU<A> relu;
    relu.set_layer({batch, oc, oh, ow});
    Layer<A>* relu_ptr = &relu;
    relu_ptr->forward(conv.output, false);

    auto* output = new UINT_TYPE[osize][DATTYPE / BITLENGTH];
    reveal_and_store(relu.output.data(), output, osize);

    float* expected = new float[osize];
    for (int b = 0; b < batch; b++)
        for (int oc_i = 0; oc_i < oc; oc_i++)
            for (int oi = 0; oi < oh; oi++)
                for (int oj = 0; oj < ow; oj++)
                {
                    float s = 0;
                    for (int ki = 0; ki < ks; ki++)
                        for (int kj = 0; kj < ks; kj++)
                        {
                            int ii = oi * stride + ki - pad, jj = oj * stride + kj - pad;
                            if (ii >= 0 && ii < ih && jj >= 0 && jj < iw)
                                s += in_f[b * ih * iw + ii * iw + jj] * ker_f[oc_i * ks * ks + ki * ks + kj];
                        }
                    expected[(b * oc + oc_i) * oh * ow + oi * ow + oj] = s > 0 ? s : 0;  // relu
                }

    int nfail = 0;
    for (int q = 0; q < osize; q++)
        for (int v = 0; v < vf; v++)
        {
            float got = FFC::ufixed_to_float(output[q][v]);
            if (got - expected[q] > epsilon || got - expected[q] < -epsilon)
            {
                nfail++;
                if (nfail <= 6)
                    print_online("conv_relu FAIL q=" + std::to_string(q) + " exp=" + std::to_string(expected[q]) +
                                 " got=" + std::to_string(got));
            }
        }
    print_online("conv_relu nfail=" + std::to_string(nfail) + " / " + std::to_string(osize));
    bool ok = (nfail == 0);
    delete[] in_f; delete[] in_v; delete[] output; delete[] expected;
    return ok;
}

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
    test_function(num_tests, num_passed, "FC+ReLU", fc_relu_test<Share>);
    test_function(num_tests, num_passed, "Conv+ReLU", conv_relu_test<Share>);

#if RESHARE_OPT == 1 && RESHARE_OPT_SIM == 1 && DATTYPE == BITLENGTH
    print_online("RESHARE_SIM reshare_b checks=" + std::to_string(g_rb_checks) +
                 " mismatch=" + std::to_string(g_rb_mismatch));
#endif
    print_stats(num_tests, num_passed);
    if (num_tests == num_passed)
        return true;
    return false;
}
