#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "bench_helper.hpp"
#include <eigen3/Eigen/Dense>

#define USE_EIGEN 1
#define FUNCTION conv_alt_bench
#define RESULTTYPE DATATYPE

template <typename T>
using MatX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T>
using VecX = Eigen::Matrix<T, Eigen::Dynamic, 1>;

#include <vector>

using namespace std;
using namespace Eigen;

int calc_outsize(int in_size, int kernel_size, int stride, int pad)
{
    return (int)std::floor((in_size + 2 * pad - kernel_size) / stride) + 1;
}

template <typename T>
T im2col_get_pixel(const T* im, int height, int width, int channels, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width)
        return T(0);  // public value, no rand needed
    return im[col + width * (row + height * channel)];
}

template <typename T>
void col2im_add_pixel(T* im, int height, int width, int channels, int row, int col, int channel, int pad, T val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width)
        return;
    im[col + width * (row + height * channel)] += val;
}

// This one might be too, can't remember.

template <typename T>
void col2im(const T* data_col, int channels, int height, int width, int ksize, int stride, int pad, T* data_im)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h)
        {
            for (w = 0; w < width_col; ++w)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                T val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad, val);
            }
        }
    }
}

// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE
template <typename T>
void im2col(const T* data_im, int channels, int height, int width, int ksize, int stride, int pad, T* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h)
        {
            for (w = 0; w < width_col; ++w)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }
}

enum class LayerType
{
    LINEAR,
    CONV2D,
    MAXPOOL2D,
    AVGPOOL2D,
    ACTIVATION,
    BATCHNORM1D,
    BATCHNORM2D,
    FLATTEN
};
template <typename T>
class Layer
{
  public:
    LayerType type;
    bool is_first;
    bool is_last;
    MatX<T> output;
    MatX<T> delta;

  public:
    Layer(LayerType type) : type(type), is_first(false), is_last(false) {}
    /* virtual void set_layer(const vector<int>& input_shape); */
    /* virtual void forward(const MatX<T>& prev_out, bool is_training = true); */
    /* virtual void backward(const MatX<T>& prev_out, MatX<T>& prev_delta); */
};

template <typename T>
class Conv2d : public Layer<T>
{
  private:
    int batch;
    int ic;
    int oc;
    int ih;
    int iw;
    int ihw;
    int oh;
    int ow;
    int ohw;
    int kh;
    int kw;
    int stride;
    int pad;
    string option;
    MatX<T> dkernel;
    VecX<T> dbias;
    MatX<T> im_col;

  public:
    MatX<T> kernel;
    VecX<T> bias;
    Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, string option);
    void set_layer(const vector<int>& input_shape);
    void forward_alt(const MatX<T>& prev_out, bool is_training);
    void backward(const MatX<T>& prev_out, MatX<T>& prev_delta);
};

template <typename T>
Conv2d<T>::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding, string option)
    : Layer<T>(LayerType::CONV2D),
      batch(0),
      ic(in_channels),
      oc(out_channels),
      ih(0),
      iw(0),
      ihw(0),
      oh(0),
      ow(0),
      ohw(0),
      kh(kernel_size),
      kw(kernel_size),
      stride(stride),
      pad(padding),
      option(option)
{
}

template <typename T>
void Conv2d<T>::set_layer(const vector<int>& input_shape)
{
    batch = input_shape[0];
    ic = input_shape[1];
    ih = input_shape[2];
    iw = input_shape[3];
    ihw = ih * iw;
    oh = calc_outsize(ih, kh, stride, pad);
    ow = calc_outsize(iw, kw, stride, pad);
    ohw = oh * ow;

    this->output.resize(batch * oc, ohw);
    this->delta.resize(batch * oc, ohw);
    kernel.resize(oc, ic * kh * kw);
    dkernel.resize(oc, ic * kh * kw);
    bias.resize(oc);
    dbias.resize(oc);
    im_col.resize(ic * kh * kw, ohw);

    int fan_in = kh * kw * ic;
    int fan_out = kh * kw * oc;
    /* init_weight(kernel, fan_in, fan_out, option); */
    /* bias.setZero(); */
}

template <typename T>
void Conv2d<T>::forward_alt(const MatX<T>& prev_out, bool is_training)
{
    for (int n = 0; n < batch; n++)
    {
        const T* im = prev_out.data() + (ic * ihw) * n;
        im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
        const int M = oc;
        const int N = ohw;
        const int K = kernel.cols();
        auto A = kernel.data();
#if FUNCTION_IDENTIFIER == 48 || FUNCTION_IDENTIFIER == 50 || FUNCTION_IDENTIFIER == 52
        auto B = im_col.data();
#else
        auto BM = im_col.transpose();
        auto B = BM.data();
#endif
        auto C = this->output.data() + (oc * ohw) * n;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                T temp = 0;
                for (int k = 0; k < K; k++)
                {
#if FUNCTION_IDENTIFIER == 48 || FUNCTION_IDENTIFIER == 50 || FUNCTION_IDENTIFIER == 52
                    temp += A[i * K + k].prepare_dot(B[k * N + j]);
#else
                    temp += A[i * K + k].prepare_dot(B[j * K + k]);
#endif
                }
                temp.mask_and_send_dot();
                C[i * N + j] = temp;
            }
        }
    }
    T::communicate();
    for (int i = 0; i < this->output.size(); i++)
    {
        this->output(i).complete_mult();
    }
    for (int n = 0; n < batch; n++)
    {
        this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
    }
}

template <typename Share>
void conv_alt_bench(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    Share::communicate();  // dummy round
    const int batch = 1;
    /* auto conv = new Conv2d<S>(64,64,3,1,"xavier_normal"); */
    /* auto conv = new Conv2d<S>(3,64,11,4,2,"xavier_normal"); */
#if FUNCTION_IDENTIFIER == 48 || FUNCTION_IDENTIFIER == 49
    auto conv = new Conv2d<S>(3, 64, 11, 4, 2, "xavier_normal");
#elif FUNCTION_IDENTIFIER == 50 || FUNCTION_IDENTIFIER == 51
    auto conv = new Conv2d<S>(3, 64, 3, 1, 1, "xavier_normal");
#elif FUNCTION_IDENTIFIER == 52 || FUNCTION_IDENTIFIER == 53
    auto conv = new Conv2d<S>(64, 64, 3, 1, 1, "xavier_normal");
#endif

    vector<int> input_shape = {batch, 64, NUM_INPUTS, NUM_INPUTS};
    MatX<S> input(batch, 64 * NUM_INPUTS * NUM_INPUTS);
    conv->set_layer(input_shape);
    conv->forward_alt(input, false);
    dummy_reveal<Share>();
}
