#pragma once
#include "../../protocols/Protocols.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <bitset>
#include "../../protocols/XOR_Share.hpp"
#include "../../protocols/Additive_Share.hpp"
#include "../../protocols/Matrix_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include "../../utils/print.hpp"

    template<typename Share>
void dummy_reveal()
{
    using S = XOR_Share<DATATYPE, Share>;
    S dummy;
    dummy.prepare_reveal_to_all();
    Share::communicate();
    dummy.complete_reveal_to_all();
}



#define USE_EIGEN 1
#if FUNCTION_IDENTIFIER == 416 || FUNCTION_IDENTIFIER == 417
#define FUNCTION conv_alt_bench
#else
#define FUNCTION conv2D_bench
#endif
#include <eigen3/Eigen/Core>

#define RESULTTYPE DATATYPE
void generateElements()
{
}

template<typename T>
using MatX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template<typename T>
using VecX = Eigen::Matrix<T, Eigen::Dynamic, 1>;


#include <vector>

using namespace std;
using namespace Eigen;

	int calc_outsize(int in_size, int kernel_size, int stride, int pad)
	{
		return (int)std::floor((in_size + 2 * pad - kernel_size) / stride) + 1;
	}

    template <typename T>
    T im2col_get_pixel(const T* im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return T(0); // public value, no rand needed
    return im[col + width * (row + height * channel)];
}

template <typename T>
void col2im_add_pixel(T* im, int height, int width, int channels,
                    int row, int col, int channel, int pad, T val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return;
    im[col + width * (row + height * channel)] += val;
}

// This one might be too, can't remember.

template <typename T>
void col2im(const T* data_col, int channels, int height, int width,
            int ksize, int stride, int pad, T* data_im)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                T val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad, val);
            }
        }
    }
}


// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE
template <typename T>
void im2col(const T* data_im, int channels, int height, int width,
            int ksize, int stride, int pad, T* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
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
template<typename T>
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
    virtual void set_layer(const vector<int>& input_shape) = 0;
    virtual void forward(const MatX<T>& prev_out, bool is_training = true) = 0;
    virtual void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) = 0;
    /* virtual void update_weight(T lr, T decay) { return; } */
    /* virtual void zero_grad() { return; } */
    /* virtual vector<int> output_shape() = 0; */
};


template<typename T> 
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
		Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
			string option);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void forward1(const MatX<T>& prev_out, bool is_training);
		void forward2(const MatX<T>& prev_out, bool is_training);
		void forward3(const MatX<T>& prev_out, bool is_training);
		void forward4(const MatX<T>& prev_out, bool is_training);
		void forward5(const MatX<T>& prev_out, bool is_training);
        void forward6(const MatX<T>& prev_out, bool is_training);
        void forward7(const MatX<T>& prev_out, bool is_training);
        void forward8(const MatX<T>& prev_out, bool is_training);
        void forward9(const MatX<T>& prev_out, bool is_training);
        void forward10(const MatX<T>& prev_out, bool is_training);
        void forward11(const MatX<T>& prev_out, bool is_training);
        void forward12(const MatX<T>& prev_out, bool is_training);
        void forward13(const MatX<T>& prev_out, bool is_training);
        void forward14(const MatX<T>& prev_out, bool is_training);
        void forward15(const MatX<T>& prev_out, bool is_training);
        void forward16(const MatX<T>& prev_out, bool is_training);
        void forward17(const MatX<T>& prev_out, bool is_training);
        void forward18(const MatX<T>& prev_out, bool is_training);
        void forward18_old(const MatX<T>& prev_out, bool is_training);
        void forward19(const MatX<T>& prev_out, bool is_training);
        void forward20(const MatX<T>& prev_out, bool is_training);
        void forward21(const MatX<T>& prev_out, bool is_training);
        void forward22(const MatX<T>& prev_out, bool is_training);
        void forward23(const MatX<T>& prev_out, bool is_training);
        void forward24(const MatX<T>& prev_out, bool is_training);
        void forward25(const MatX<T>& prev_out, bool is_training);
        void forward26(const MatX<T>& prev_out, bool is_training);
        void forward27(const MatX<T>& prev_out, bool is_training);
        void forward28(const MatX<T>& prev_out, bool is_training);
        void forward29(const MatX<T>& prev_out, bool is_training);
        void forward30(const MatX<T>& prev_out, bool is_training);
        void forward31(const MatX<T>& prev_out, bool is_training);
        void forward32(const MatX<T>& prev_out, bool is_training);
        void forward33(const MatX<T>& prev_out, bool is_training);
        void forward34(const MatX<T>& prev_out, bool is_training);
        void forward_alt(const MatX<T>& prev_out, bool is_training);
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		/* void update_weight(T lr, T decay) override; */
		/* void zero_grad() override; */
		/* vector<int> output_shape() override; */
	};

    template<typename T>
	Conv2d<T>::Conv2d(
		int in_channels,
		int out_channels,
		int kernel_size,
        int stride,
		int padding,
		string option
	) :
		Layer<T>(LayerType::CONV2D),
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
		option(option) {}

    template<typename T>
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


int TILE_SIZE = 64;
    template<typename T>
	void Conv2d<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
			this->output.block(oc * n, 0, oc, ohw).noalias() = kernel * im_col;
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
		}
	}
   
//Eigen
    template<typename T>
	void Conv2d<T>::forward1(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
			this->output.block(oc*n,0, oc, ohw).noalias() = kernel * im_col;
        }
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).mask_and_send_dot();
            }
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
		
	}
    
    template<typename T>
	void Conv2d<T>::forward2(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            for(int i = 0; i < oc; ++i) {
        for(int k = 0; k < kernel.cols(); ++k) {
            T temp = kernel(i, k);
                for(int j = 0; j < ohw; ++j) {
                    this->output(oc * n + i, j) += temp * im_col(k, j);  // Use custom * and + operators
                    }
                }
                for(int j = 0; j < ohw; ++j) 
                    this->output(oc * n + i, j).mask_and_send_dot();
        }
        }
        /* for (int j = 0; j < this->output.size(); j++) { */
        /*     this->output(j).mask_and_send_dot(); */
        /* } */

            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
	}
    
            template<typename T>
	void Conv2d<T>::forward3(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
        for(int k = 0; k < kernel.cols(); ++k) {
            for(int i = 0; i < oc; ++i) {
                for(int j = 0; j < ohw; ++j) {
                    this->output(oc *n + i, j) += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators
                    }
                }
        }
        }
        for (int j = 0; j < this->output.size(); j++) {
            this->output(j).mask_and_send_dot();
        }

            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
	}
      // naive      
            template<typename T>
	void Conv2d<T>::forward4(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            for(int i = 0; i < oc; ++i) {
                T sum = T(0);
                for(int j = 0; j < ohw; ++j) {
                        for(int k = 0; k < kernel.cols(); ++k) {
                    sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators
                    }
                sum.mask_and_send_dot();
                this->output(oc * n + i, j) = sum;
                }
        }
    }

            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
	}
            
            template<typename T>
	void Conv2d<T>::forward5(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
    auto A = kernel;
    auto B = im_col;
    auto C = this->output;
    const int Mtile = 64; // tile size in M dimension
    const int Ntile = 64; // tile size in N dimension
    const int M = oc;
    const int N = ohw; 
    const int K = kernel.cols(); 

    for (int m = 0; m < M; m += Mtile)                // iterate over M dimension
    {
    for (int q = 0; q < N; q += Ntile)            // iterate over N dimension
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < Mtile; ++i)       // compute one tile 
                for (int j = 0; j < Ntile; ++j) {
                    int row = m + i;
                    int col = q + j;
                    C(n*oc + row,col) += A(row,k) * B(k,col);
                    /* C[row][col] += A[row][k] * B[k][col]; */
                }
    }
        }
    
        for (int i = 0; i < this->output.size(); i++) {
            this->output(i).mask_and_send_dot();
    }
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
            template<typename T>
	void Conv2d<T>::forward6(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
    auto A = kernel;
    auto B = im_col;
    auto C = this->output;
    const int Mtile = 64; // tile size in M dimension
    const int Ntile = 64; // tile size in N dimension
    const int M = oc;
    const int N = ohw; 
    const int K = kernel.cols(); 

    for (int m = 0; m < M; m += Mtile)                // iterate over M dimension
    {
    for (int q = 0; q < N; q += Ntile)            // iterate over N dimension
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < Mtile; ++i)       // compute one tile 
                    {
                    int row = m + i;
                    T temp = A(row,k);
                for (int j = 0; j < Ntile; ++j) {
                    int col = q + j;
                    C(n*oc + row,col) += temp * B(k,col);
                    /* C[row][col] += A[row][k] * B[k][col]; */
                }
                }
    }
        }
    
        for (int i = 0; i < this->output.size(); i++) {
            this->output(i).mask_and_send_dot();
    }
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

            template<typename T>
	void Conv2d<T>::forward7(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
    auto A = kernel;
    auto B = im_col;
    auto C = this->output;
    const int Mtile = 64; // tile size in M dimension
    const int Ntile = 64; // tile size in N dimension
    const int M = oc;
    const int N = ohw; 
    const int K = kernel.cols(); 

    for (int m = 0; m < M; m += Mtile)                // iterate over M dimension
    {
    for (int q = 0; q < N; q += Ntile)            // iterate over N dimension
        for (int k = 0; k < K; ++k)
                for (int j = 0; j < Ntile; ++j) {
                    int col = q + j;
                    T temp = B(k,col);
            for (int i = 0; i < Mtile; ++i)       // compute one tile 
                    {
                    int row = m + i;
                    C(n*oc + row,col) += A(row,k) * temp;
                    /* C[row][col] += A[row][k] * B[k][col]; */
                }
                }
    }
        }
    
        for (int i = 0; i < this->output.size(); i++) {
            this->output(i).mask_and_send_dot();
    }
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
            
    template<typename T>
	void Conv2d<T>::forward8(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */
    auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    /* const int m = kernel.rows(); */
    /* const int f = kernel.cols(); */
    /* const int p = im_col.cols(); */
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;

  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    for (int jj = j; jj < j_max; ++jj) {
                        for (int kk = k; kk < k_max; ++kk) {
                            /* C(oc * n + ii,jj) += A(ii,kk) * B(kk,jj); */
                            /* C[ii][jj] += A[ii][kk] * B[kk][jj]; */
                       C[ii * p + jj] += A[ii * f + kk] * B[kk * p + jj]; 
                        }
                    }
                }
            }
                for (int ii = i; ii < i_max; ++ii) 
                    for (int jj = j; jj < j_max; ++jj) 
                        C[ii * p + jj].mask_and_send_dot();
    }
}
} 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    
    template<typename T>
	void Conv2d<T>::forward9(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
                for (int ii = i; ii < i_max; ++ii) {
                    for (int jj = j; jj < j_max; ++jj) {
                        for (int k = 0; k < f; k++) {
                /* int k_max = std::min(k + TILE_SIZE, f); */
                /*         for (int kk = k; kk < k_max; ++kk) { */
                            /* for (int k = 0; k < f; k ++) { */
                       C[ii * p + jj] += A[ii * f + k] * B[k * p + jj]; 
                        }
                    C[ii * p + jj].mask_and_send_dot();
                    }
                }
            /* for (int ii = i; ii < i_max; ++ii) { */
            /*     for (int jj = j; jj < j_max; ++jj) { */
            /*         C[ii * p + jj].mask_and_send_dot(); */
            /*     } */
            /* } */
            }
        }
    




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward10(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */
    auto A = kernel.data();
    /* auto B = im_col.transpose().data(); */
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    T temp = A[ii*f+kk]; 
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii * p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                for (int jj = j; jj < j_max; ++jj) {
                    C[ii * p + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    
    template<typename T>
	void Conv2d<T>::forward11(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                        for (int kk = k; kk < k_max; ++kk) {
                    const int row2 = kk*p;
                for (int ii = i; ii < i_max; ++ii) {
                    const int row = ii*p;
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[row + jj] += temp * B[row2 + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    
    template<typename T>
	void Conv2d<T>::forward12(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
          for (int i = 0; i < n; i ++) {
                    for (int jj = j; jj < j_max; ++jj) {
                        T temp = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                    temp += A[i*f+kk] * B[kk*p + jj]; 
                        }
                        temp.mask_and_send_dot();
                       C[i * p + jj] += temp; 
    }
  }
            }
        }
        } 




        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward13(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
       
for (int i = 0; i < m; i += TILE_SIZE) {
    int i_max = std::min(i + TILE_SIZE, m);
    for (int k = 0; k < f; k += TILE_SIZE) {
        int k_max = std::min(k + TILE_SIZE, f);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);

            for (int ii = i; ii < i_max; ++ii) {
                for (int kk = k; kk < k_max; ++kk) {
                    auto temp = A[ii * f + kk];
                    for (int jj = j; jj < j_max; ++jj) {
                        C[ii * p + jj] += temp * B[kk * p + jj];
                    }
                }
            }
        }
    }
}

// Post-processing step
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
        C[i * p + j].mask_and_send_dot();
    }
}

}
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }


    template<typename T>
	void Conv2d<T>::forward14(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
        

for (int i0 = 0; i0 < m; i0 += TILE_SIZE) {
    int i_max = std::min(i0 + TILE_SIZE, m);
    for (int k0 = 0; k0 < f; k0 += TILE_SIZE) {
        int k_max = std::min(k0 + TILE_SIZE, f);
        for (int j0 = 0; j0 < p; j0 += TILE_SIZE) {
            int j_max = std::min(j0 + TILE_SIZE, p);

            for (int i = i0; i < i_max; ++i) {
                for (int j = j0; j < j_max; ++j) {
                    T temp_sum = T(0);
                    for (int k = k0; k < k_max; ++k) {
                        temp_sum += A[i * f + k] * B[k * p + j];
                    }
                    C[i * p + j] += temp_sum;
                }
            }
        }
    }
}

// Post-processing step
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
        C[i * p + j].mask_and_send_dot();
    }
}





        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    
    template<typename T>
	void Conv2d<T>::forward_alt(const MatX<T>& prev_out, bool is_training)
	{
		for (int n = 0; n < batch; n++) {
        const T* im = prev_out.data() + (ic * ihw) * n;
        im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
        const int M = oc;
        const int N = ohw; 
        const int K = kernel.cols(); 
            auto A = kernel.data();
#if FUNCTION_IDENTIFIER == 416
            auto B = im_col.data();
#else
            auto BM = im_col.transpose();
            auto B = BM.data();
#endif
            auto C = this->output.data() + (oc * ohw) * n;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    T temp = 0;
                    for (int k = 0; k < K; k++) {
#if FUNCTION_IDENTIFIER == 416
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
        for (int i = 0; i < this->output.size(); i++) {
            this->output(i).complete_mult();
        }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
        
    }
    
    template<typename T>
	void Conv2d<T>::forward15(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    const int row = ii*p;
                        for (int kk = k; kk < k_max; ++kk) {
                    const int row2 = kk*p;
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[row + jj] += temp * B[row2 + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    template<typename T>
	void Conv2d<T>::forward16(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                        for (int kk = k; kk < k_max; ++kk) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

        

    template<typename T>
	void Conv2d<T>::forward17(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    // all opt
    template<typename T>
	void Conv2d<T>::forward18(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */
            auto A = kernel.data();
    /* auto B = im_col.transpose().data(); */
        auto BM = im_col.transpose();
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
      /* _mm_prefetch(A + i * f, _MM_HINT_T0); */
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            C[i * p + j] = T(0);
            /* _mm_prefetch(B + j * f, _MM_HINT_T0); */
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    const int iif = ii*f;
                    const int iip = ii*p;

                    /* const int row2 = ii*f+kk; */
                    for (int jj = j; jj < j_max; ++jj) {
                    auto temp = T(0);
                        const int jjf = jj*f;
                        for (int kk = k; kk < k_max; ++kk) {
                            /* _mm_prefetch(C + ii * p + jj, _MM_HINT_T0); */
                       temp += A[iif+kk] * B[jjf + kk]; 
                        }
                        C[iip + jj] += temp;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }

        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            /* T::communicate(); */
            /* for (int i = 0; i < this->output.size(); i++) { */
            /*     this->output(i).complete_mult(); */
            /* } */
        T::communicate();
            /* auto b = bias.data(); */
            const int m = oc;
            const int p = ohw;
		for (int n = 0; n < batch; n++) {
            auto C = this->output.data() + (oc * ohw) * n;
            for( int i = 0; i < m; i += TILE_SIZE) {
                const int i_max = std::min(i + TILE_SIZE, m);
                for (int j = 0; j < p; j += TILE_SIZE) {
                    const int j_max = std::min(j + TILE_SIZE, p);
                    for (int ii = i; ii < i_max; ++ii) {
                        const int row = ii*p;
                        for (int jj = j; jj < j_max; ++jj) {
                            C[row + jj].complete_mult();
                            /* C[row + jj] += b[row+jj]; */
                        }
                    }
                }
            }
    }
		for (int n = 0; n < batch; n++)
            this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
    }
    template<typename T>
	void Conv2d<T>::forward18_old(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */
            auto A = kernel.data();
    auto BM = im_col.transpose();
        /* MatX<T> BM = im_col.transpose(); */
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
      /* _mm_prefetch(A + i * f, _MM_HINT_T0); */
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            /* _mm_prefetch(B + j * f, _MM_HINT_T0); */
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    for (int jj = j; jj < j_max; ++jj) {
                    auto temp = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                            /* _mm_prefetch(C + ii * p + jj, _MM_HINT_T0); */
                       temp += A[ii*f+kk] * B[jj*f + kk]; 
                        }
                        C[ii*p + jj] += temp;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }

        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward19(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */
            auto A = kernel.data();
    auto B = im_col.transpose().data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = T(0);
                    for (int jj = j; jj < j_max; ++jj) {
                        for (int k = 0; k < f; k ++) {
                       temp += A[ii*f+k] * B[jj*f + k]; 
                        }
                        temp.mask_and_send_dot();
                        C[ii*p + jj] = temp;
                    }
                }
            }
            }

        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    //transpose
    template<typename T>
	void Conv2d<T>::forward20(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    /* auto B = im_col.transpose().data(); */
    auto BM = im_col.transpose();
        /* MatX<T> BM = im_col.transpose(); */
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i ++) {
        for (int j = 0; j < p; j ++) {
                    auto temp = T(0);
                        for (int k = 0; k < f; k ++) {
                       temp += A[i*f+k] * B[j*f + k]; 
                        }
                        temp.mask_and_send_dot();
                        C[i*p + j] = temp;
                    }
                }
            }
  /* for (int i = 0; i < m; i += TILE_SIZE) { */
  /*       int i_max = std::min(i + TILE_SIZE, m); */
  /*       for (int j = 0; j < p; j += TILE_SIZE) { */
  /*           int j_max = std::min(j + TILE_SIZE, p); */
  /*           for (int k = 0; k < f; k += TILE_SIZE) { */
  /*               int k_max = std::min(k + TILE_SIZE, f); */
  /*               for (int ii = i; ii < i_max; ++ii) { */
  /*                       for (int kk = k; kk < k_max; ++kk) { */
  /*                   /1* const int row2 = ii*f+kk; *1/ */
  /*                   auto temp = A[ii*f+kk]; */
  /*                   for (int jj = j; jj < j_max; ++jj) { */
  /*                      C[ii*p + jj] += temp * B[jj*f + kk]; */ 
  /*                       } */
  /*                   } */
  /*               } */
  /*           } */
  /*           for (int ii = i; ii < i_max; ++ii) { */
  /*               const int row = ii*p; */
  /*               for (int jj = j; jj < j_max; ++jj) { */
  /*                   C[row + jj].mask_and_send_dot(); */
  /*               } */
  /*           } */
  /*       } */
  /*   } */
        




        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward21(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.transpose().data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    for (int jj = j; jj < j_max; ++jj) {
                        for (int kk = k; kk < k_max; ++kk) {
                       C[ii*p + jj] += A[ii*f+kk] * B[jj*f + kk]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward22(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    MatX<T> BM = im_col.transpose();
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    for (int jj = j; jj < j_max; ++jj) {
                        T temp_sum = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                       temp_sum += A[ii*f+kk] * B[jj*f + kk]; 
                        }
                        C[ii*p + jj] += temp_sum;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward23(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    MatX<T> BM = im_col.transpose();
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                    for (int jj = j; jj < j_max; ++jj) {
                for (int ii = i; ii < i_max; ++ii) {
                    int row = ii*f+k;
                    int row2 = jj*f+k;
                    /* const int row2 = ii*f+kk; */
                        T temp_sum = T(0);
                        while(row < ii*f+k_max){
                       temp_sum += A[row++] * B[row2++]; 
                        }
                        C[ii*p + jj] += temp_sum;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                int row = ii*p+j;
                while(row < ii*p+j_max){
                    C[row++].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward24(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    MatX<T> BM = im_col.transpose();
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                    for (int jj = j; jj < j_max; ++jj) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                        T temp_sum = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                       temp_sum += A[ii*f+kk] * B[jj*f + kk]; 
                        }
                        C[ii*p + jj] += temp_sum;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                for (int jj = j; jj < j_max; ++jj) {
                    C[ii*p + jj].mask_and_send_dot();
                }
            }
        }
    }
        

        }


        
    
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward25(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            C[i*p + j] = T(0);
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward26(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
  for (int i = 0; i < m; i += TILE_SIZE) {
      C[i*p + j] = T(0);
        int i_max = std::min(i + TILE_SIZE, m);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward27(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
        for( int i = 0; i < this->output.size(); ++i) {
            this->output(i) = T(0);
        }
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
        }
    }

            for (int i = 0; i < m*p; ++i) {
                this->output(i).mask_and_send_dot();
            }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    template<typename T>
	void Conv2d<T>::forward28(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
        for( int i = 0; i < this->output.size(); ++i) {
            this->output(i) = T(0);
        }
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
        }
    }

            for (int i = 0; i < m*p; ++i) {
                this->output(i).mask_and_send_dot();
            }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    template<typename T>
	void Conv2d<T>::forward29(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
        for( int i = 0; i < this->output.size(); ++i) {
            this->output(i) = T(0);
        }
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
                for (int ii = i; ii < i_max; ++ii) {
                        for (int kk = k; kk < k_max; ++kk) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
        }
    }

            for (int i = 0; i < m*p; ++i) {
                this->output(i).mask_and_send_dot();
            }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    template<typename T>
	void Conv2d<T>::forward30(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
        for( int i = 0; i < this->output.size(); ++i) {
            this->output(i) = T(0);
        }
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    for (int jj = j; jj < j_max; ++jj) {
                    auto temp = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                       temp += A[ii*f+kk] * B[kk*p + jj]; 
                        }
                    C[ii*p + jj] += temp;
                    }
                }
            }
        }
    }

            for (int i = 0; i < m*p; ++i) {
                this->output(i).mask_and_send_dot();
            }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    
    template<typename T>
	void Conv2d<T>::forward31(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
        for( int i = 0; i < this->output.size(); ++i) {
            this->output(i) = T(0);
        }
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                    for (int jj = j; jj < j_max; ++jj) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                       temp += A[ii*f+kk] * B[kk*p + jj]; 
                        }
                    C[ii*p + jj] += temp;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }

        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }


    template<typename T>
	void Conv2d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, stride, pad, im_col.data());
			dkernel += this->delta.block(oc * n, 0, oc, ohw) * im_col.transpose();
			dbias += this->delta.block(oc * n, 0, oc, ohw).rowwise().sum();
		}

		if (!this->is_first) {
			for (int n = 0; n < batch; n++) {
				T* begin = prev_delta.data() + ic * ihw * n;
				im_col = kernel.transpose() * this->delta.block(oc * n, 0, oc, ohw);
				col2im(im_col.data(), ic, ih, iw, kh, stride, pad, begin);
			}
		}
	}

template <typename T>
class SH{
DATATYPE s1;
DATATYPE s2;
    public:



static SH get_S(UINT_TYPE val){
SH s;
UINT_TYPE s_arr[sizeof(T)/sizeof(UINT_TYPE)] = {val};
UINT_TYPE r_arr[sizeof(T)/sizeof(UINT_TYPE)];
for(int i = 0; i < sizeof(T)/sizeof(UINT_TYPE); i++){
    r_arr[i] = rand();
}
orthogonalize_arithmetic(s_arr, &s.s1 , 1);
orthogonalize_arithmetic(r_arr, &s.s2 , 1);
s.s1 = OP_SUB(s.s1, s.s2);
return s;
}

SH(UINT_TYPE val){
UINT_TYPE s_arr[sizeof(T)/sizeof(UINT_TYPE)] = {val};
orthogonalize_arithmetic(s_arr, &s1 , 1);
s2 = SET_ALL_ZERO();
}



SH(T s1, T s2){
this->s1 = s1;
this->s2 = s2;
}


SH(){
this->s1 = SET_ALL_ZERO();
this->s2 = SET_ALL_ZERO();
}


SH operator+(const SH s) const{
    return SH(this->s1 + s.s1, this->s2 + s.s2);
}

SH operator-(const SH s) const{
    return SH(this->s1 - s.s1, this->s2 - s.s2);
}

SH operator*(const SH s) const{
    auto ls1 = OP_ADD( OP_MULT(this->s1, s.s1), OP_MULT(this->s1, s.s2));
    auto ls2 = OP_ADD( OP_MULT(this->s2, s.s1), OP_MULT(this->s2, s.s2));
    return SH(ls1, ls2);
}

SH operator*(const UINT_TYPE s) const{
    return SH(OP_MULT(s1, PROMOTE(s)), OP_MULT(s2, PROMOTE(s)));
}

SH operator/(const UINT_TYPE s) const{
    /* return SH(OP_DIV(s1, PROMOTE(s)), OP_DIV(s2, PROMOTE(s))); */ // not supported for now
    return SH();
}

void operator+=(const SH s){
    this->s1 += s.s1;
    this->s2 += s.s2;
}

void operator-=(const SH s){
    this->s1 -= s.s1;
    this->s2 -= s.s2;
}

void operator*= (const SH s){
    this->s1 = OP_ADD( OP_MULT(this->s1, s.s1), OP_MULT(this->s1, s.s2));
    this->s2 = OP_ADD( OP_MULT(this->s2, s.s1), OP_MULT(this->s2, s.s2));
}


//needed for Eigen optimization
bool operator==(const SH& other) const {
    return false; 
}

SH trunc_local() const{
    return SH(OP_TRUNC(s1), OP_TRUNC(s2));
}

template<typename float_type, int fractional>
float_type reveal_float() const{

    UINT_TYPE s_arr[sizeof(T)/sizeof(UINT_TYPE)];
    T temp = OP_ADD(s1, s2);
    unorthogonalize_arithmetic(&temp, s_arr, 1);
    float_type result = fixedToFloat<float_type, UINT_TYPE, fractional>(s_arr[0]);
    return result;
    }


};

template<typename T>
SH<T> truncate(const SH<T>& val) {
    return val.trunc_local();
}




    template<typename Share>
void forward_pass(DATATYPE* res)
{
Share::communicate(); // Dummy communication round to simulate input sharing
using D = sint_t<Matrix_Share<DATATYPE, Share>>;
/* using D = Matrix_Share<DATATYPE, Share>; */
/* using M = SH<DATATYPE>; */
/* using D = SH<DATATYPE>; */
/* Conv2d<M> conv(3,64,3,1); */
#if FUNCTION_IDENTIFIER == 20
std::vector<int> input_shape = {1, 3, NUM_INPUTS, NUM_INPUTS};
MatX<D> input(1, NUM_INPUTS * NUM_INPUTS * 3);
Conv2d<D> d_conv(3, 64, 3, 1, "xavier_normal");
#elif FUNCTION_IDENTIFIER == 23
Conv2d<D> d_conv(64, 64, 3, 1, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method)
vector<int> input_shape = {1, 64, NUM_INPUTS, NUM_INPUTS};
MatX<D> input(1, 64 * NUM_INPUTS * NUM_INPUTS);
#else
Conv2d<D> d_conv(64, 128, 3, 1, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method)
vector<int> input_shape = {1, 64, NUM_INPUTS/2, NUM_INPUTS/2};
MatX<D> input(1, 64 * NUM_INPUTS/2 * NUM_INPUTS/2);
#endif
    d_conv.set_layer(input_shape);
    for (int j = 0; j < d_conv.output.size(); j++) {
    d_conv.output(j).mask_and_send_dot();
    }
    Share::communicate();
    for (int j = 0; j < d_conv.output.size(); j++) {
    d_conv.output(j).complete_mult();
    }

    //dummy reveal
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    UINT_TYPE dummy[DATTYPE];
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);


}
template<typename Share>
void cryptgpu_figure1a(DATATYPE* res)
{
using D = Matrix_Share<DATATYPE, Share>;
const int batch = 1;
Conv2d<D> d_conv(3, 64, 11, 4, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method)
vector<int> input_shape = {batch, 64, NUM_INPUTS, NUM_INPUTS};
MatX<D> input(batch, 64 * NUM_INPUTS * NUM_INPUTS);

}
    
    template<typename Share>
void conv2D_bench(DATATYPE* res)
{
Share::communicate(); // Dummy communication round to simulate input sharing
/* using D = Additive_Share<DATATYPE, Share>; */
using D = Matrix_Share<DATATYPE, Share>;
const int batch = 1;
/* using D = Matrix_Share<DATATYPE, Share>; */
/* using M = SH<DATATYPE>; */
/* using D = SH<DATATYPE>; */
/* Conv2d<M> conv(3,64,3,1); */
Conv2d<D> d_conv(64, 64, 3, 1, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method)
vector<int> input_shape = {batch, 64, NUM_INPUTS, NUM_INPUTS};
MatX<D> input(batch, 64 * NUM_INPUTS * NUM_INPUTS);
    d_conv.set_layer(input_shape);
    alignas(sizeof(DATATYPE))UINT_TYPE dummy[DATTYPE];
#if FUNCTION_IDENTIFIER == 56
d_conv.forward4(input, false);
#elif FUNCTION_IDENTIFIER == 57
d_conv.forward20(input, false);
#elif FUNCTION_IDENTIFIER == 37
d_conv.forward1(input, false);
#elif FUNCTION_IDENTIFIER == 38
d_conv.forward10(input, false);
#elif FUNCTION_IDENTIFIER == 39
d_conv.forward11(input, false);
#elif FUNCTION_IDENTIFIER == 40
d_conv.forward17(input, false);
#elif FUNCTION_IDENTIFIER == 41
d_conv.forward18(input, false);
#elif FUNCTION_IDENTIFIER == 42
d_conv.forward21(input, false);
#elif FUNCTION_IDENTIFIER == 43
d_conv.forward26(input, false);
#elif FUNCTION_IDENTIFIER == 44
d_conv.forward21(input, false);
#elif FUNCTION_IDENTIFIER == 45
d_conv.forward9(input, false);
#elif FUNCTION_IDENTIFIER == 46
d_conv.forward10(input, false);
#elif FUNCTION_IDENTIFIER == 47
d_conv.forward11(input, false);
#elif FUNCTION_IDENTIFIER == 48
d_conv.forward12(input, false);
#elif FUNCTION_IDENTIFIER == 49
d_conv.forward13(input, false);
#elif FUNCTION_IDENTIFIER == 50
d_conv.forward14(input, false);
#elif FUNCTION_IDENTIFIER == 51
d_conv.forward23(input, false);
#elif FUNCTION_IDENTIFIER == 52
d_conv.forward16(input, false);
#elif FUNCTION_IDENTIFIER == 53
d_conv.forward17(input, false);
#elif FUNCTION_IDENTIFIER == 54
std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward10(input, false);
std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 10: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward1(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 1: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward2(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 2: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward3(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 3: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward4(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 4: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward5(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 5: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward6(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 6: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward7(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 7: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward8(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 8: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward9(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken 9: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward11(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 11: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward12(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 12: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward13(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 13: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward14(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 14: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward15(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 15: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward16(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 16: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward17(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 17: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward18(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 18: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward19(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 19: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward20(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 20: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward21(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken 21: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward22(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 22: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward23(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 23: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward24(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 24: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward25(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 25: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward26(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 26: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward27(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 27: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward28(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 28: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward29(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 29: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward30(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 30: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward31(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 31: " << duration << std::endl;
#elif FUNCTION_IDENTIFIER == 55
TILE_SIZE = 8;
std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward17(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 17, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 17: " << duration << std::endl;


t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward18(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 18, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 18: " << duration << std::endl;

t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward18_old(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 18 (old), TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 18: " << duration << std::endl;



t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward11(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 11, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;


t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward12(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 12, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 12: " << duration << std::endl;


t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 11: " << duration << std::endl;


t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward10(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 10, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 10: " << duration << std::endl;

while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward12(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 12, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 12: " << duration << std::endl;

while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward13(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 13, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 13: " << duration << std::endl;

while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward15(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 15, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 15: " << duration << std::endl;

while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward16(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 16, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 16: " << duration << std::endl;

while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward21(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 21, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 21: " << duration << std::endl;


while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward26(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 26, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 26: " << duration << std::endl;


#endif
    //dummy reveal
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);


}


/* template<typename Share> */
/* void backward_pass(DATATYPE* res) */
/* { */

/* Share::communicate(); // Dummy communication round to simulate input sharing */
/* using D = sint_t<Matrix_Share<DATATYPE, Share>>; */
/* /1* using D = Matrix_Share<DATATYPE, Share>; *1/ */
/* #if FUNCTION_IDENTIFIER == 21 */ 
/* std::vector<int> input_shape = {1, 3, NUM_INPUTS, NUM_INPUTS}; */
/* MatX<D> input(1, NUM_INPUTS * NUM_INPUTS * 3); */
/* Conv2d<D> d_conv(3, 64, 3, 1, "xavier_normal"); */
/* #elif FUNCTION_IDENTIFIER == 24 */
/* Conv2d<D> d_conv(64, 64, 3, 1, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method) */
/* vector<int> input_shape = {1, 64, NUM_INPUTS, NUM_INPUTS}; */
/* MatX<D> input(1, 64 * NUM_INPUTS * NUM_INPUTS); */
/* #else */
/* Conv2d<D> d_conv(64, 128, 3, 1, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method) */
/* vector<int> input_shape = {1, 64, NUM_INPUTS/2, NUM_INPUTS/2}; */
/* MatX<D> input(1, 64 * NUM_INPUTS/2 * NUM_INPUTS/2); */
/* #endif */
/* d_conv.set_layer(input_shape); */

/* /1* conv.set_layer(input_shape); *1/ */
/* d_conv.backward(input,d_conv.output); */
/*     for (int j = 0; j < d_conv.output.size(); j++) { */
/*     d_conv.output(j).mask_and_send_dot(); */
/*     } */
/*     Share::communicate(); */
/*     for (int j = 0; j < d_conv.output.size(); j++) { */
/*     d_conv.output(j).complete_mult(); */
/*     } */
    
/*     //dummy reveal */
/*     d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all(); */
/*     Share::communicate(); */
/*     UINT_TYPE dummy[DATTYPE]; */
/*     d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy); */

/* } */


    template<typename Share>
void FC_bench(DATATYPE* res)
{
    Share::communicate(); // Dummy communication round to simulate input sharing
    using S = sint_t<Matrix_Share<DATATYPE, Share>>;
    /* using M = Matrix_Share<DATATYPE, Share>; */
    VecX<S> a(NUM_INPUTS);
    VecX<S> c(NUM_INPUTS);
    MatX<S> b(NUM_INPUTS, NUM_INPUTS);
    c = b * a;
    
    for(int i = 0; i < NUM_INPUTS; i++)
    {
            c(i).mask_and_send_dot();
    }

    Share::communicate();
    
    for(int i = 0; i < NUM_INPUTS; i++)
    {
            c(i).complete_mult();
    }
    
    //dummy reveal
    c(NUM_INPUTS - 1).prepare_reveal_to_all();
    Share::communicate();
    UINT_TYPE dummy[DATTYPE];
    c(NUM_INPUTS - 1).complete_reveal_to_all(dummy);
}

template<typename Share>
void dot_prod_eigen_bench(DATATYPE* res)
{
    Share::communicate(); // Dummy communication round to simulate input sharing
    using S = Matrix_Share<DATATYPE, Share>;
    VecX<S> a(NUM_INPUTS);
    VecX<S> c(NUM_INPUTS);
    MatX<S> b(NUM_INPUTS, NUM_INPUTS);
    c = b * a;
    
    for(int i = 0; i < NUM_INPUTS; i++)
    {
            c(i).mask_and_send_dot();
    }

    Share::communicate();
    
    for(int i = 0; i < NUM_INPUTS; i++)
    {
            c(i).complete_mult();
    }

    //dummy reveal
    c(NUM_INPUTS - 1).prepare_reveal_to_all();
    Share::communicate();
    UINT_TYPE dummy[DATTYPE];
    c(NUM_INPUTS - 1).complete_reveal_to_all(dummy);
}

template<typename Share>
void conv_alt_bench(DATATYPE* res)
{
    using S = Additive_Share<DATATYPE, Share>;
    Share::communicate(); // dummy round
    const int batch = 1;
    /* auto conv = new Conv2d<S>(64,64,3,1,"xavier_normal"); */
    auto conv = new Conv2d<S>(3,64,11,4,2,"xavier_normal");
    vector<int> input_shape = {batch, 64, NUM_INPUTS, NUM_INPUTS};
    MatX<S> input(batch, 64 * NUM_INPUTS * NUM_INPUTS);
    conv->set_layer(input_shape);
    conv->forward_alt(input, false);
    dummy_reveal<Share>();
}
