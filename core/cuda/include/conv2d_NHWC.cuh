#pragma once

#include <cuda_runtime.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/conv/kernel/default_conv2d_dgrad.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_wgrad.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/device_memory.h>
#include <thrust/device_vector.h>
#include <utils.cuh>
#include <math.h>
#include <stdlib.h>

template <typename T>
inline cutlass::TensorRef<T, cutlass::layout::TensorNHWC> toTensorRef(T* ptr, int n, int h, int w, int c)
{

    return cutlass::TensorRef<T, cutlass::layout::TensorNHWC>(ptr, cutlass::layout::TensorNHWC::packed({n, h, w, c}));
}

template <typename T>
using Conv2dFpropKernel =
    typename cutlass::conv::kernel::DefaultConv2dFprop<T,
                                                       cutlass::layout::TensorNHWC,
                                                       T,
                                                       cutlass::layout::TensorNHWC,
                                                       T,
                                                       cutlass::layout::TensorNHWC,
                                                       T,
                                                       cutlass::arch::OpClassSimt,
                                                       cutlass::arch::Sm80,
                                                       cutlass::gemm::GemmShape<32, 64, 8>,
                                                       cutlass::gemm::GemmShape<32, 64, 8>,
                                                       cutlass::gemm::GemmShape<1, 1, 1>,
                                                       cutlass::epilogue::thread::LinearCombination<T, 1, T, T>,
                                                       cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
                                                       4,
                                                       cutlass::arch::OpMultiplyAdd,
                                                       cutlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;

template <typename T>
using FpropImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel<T>>;

template <typename T>
struct FpropOptions
{

    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;
    T alpha;
    T beta;

    FpropOptions(int in_n,
                 int in_h,
                 int in_w,
                 int in_c,
                 int f_k,
                 int f_r,
                 int f_s,
                 int f_c,
                 int padding_h,
                 int padding_w,
                 int _stride,
                 int _dilation,
                 T _alpha,
                 T _beta)
        : input_size(in_n, in_h, in_w, in_c),
          filter_size(f_k, f_r, f_s, f_c),
          padding(padding_h, padding_h, padding_w, padding_w),
          conv_stride(_stride, _stride),
          dilation(_dilation, _dilation),
          alpha(_alpha),
          beta(_beta)
    {
    }

    cutlass::Tensor4DCoord output_size() const
    {
        return cutlass::Tensor4DCoord(
            input_size.n(),
            (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
            (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
            filter_size.n());
    }
};

template <typename T>
cudaError_t CutlassConvFprop(const cutlass::TensorRef<T, cutlass::layout::TensorNHWC>& A,
                             const cutlass::TensorRef<T, cutlass::layout::TensorNHWC>& B,
                             cutlass::TensorRef<T, cutlass::layout::TensorNHWC>& C,
                             FpropOptions<T> const& options)
{

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::conv::Conv2dProblemSize problem_size(options.input_size,
                                                  options.filter_size,
                                                  options.padding,
                                                  options.conv_stride,
                                                  options.dilation,
                                                  options.output_size(),
                                                  mode,
                                                  1  // split_k_slices
    );

    typename FpropImplicitGemm<T>::Arguments arguments{problem_size, A, B, C, C, {options.alpha, options.beta}};

    FpropImplicitGemm<T> implicit_gemm_op;

    size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // memory_profiler.track_alloc(workspace_size);

    auto status = implicit_gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = implicit_gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    status = implicit_gemm_op();
    CUTLASS_CHECK(status);

    // memory_profiler.track_free(workspace_size);

    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

namespace gpu
{

template <typename T>
void conv_fprop(cutlass::TensorRef<T, cutlass::layout::TensorNHWC> A,
                cutlass::TensorRef<T, cutlass::layout::TensorNHWC> B,
                cutlass::TensorRef<T, cutlass::layout::TensorNHWC> C,
                int b,
                int imageHeight,
                int imageWidth,
                int Din,
                int Dout,
                int filterHeight,
                int filterWidth,
                int paddingHeight,
                int paddingWidth,
                int stride,
                int dilation)
{

    FpropOptions<T> options(b,
                            imageHeight,
                            imageWidth,
                            Din,
                            Dout,
                            filterHeight,
                            filterWidth,
                            Din,
                            paddingHeight,
                            paddingWidth,
                            stride,
                            dilation,
                            (T)1,
                            (T)0);

    CutlassConvFprop<T>(A, B, C, options);
    cudaDeviceSynchronize();
}

template <typename T>
void conv_fprop(T* A_ptr,
                T* B_ptr,
                T* C_ptr,
                int b,
                int imageHeight,
                int imageWidth,
                int Din,
                int Dout,
                int filterHeight,
                int filterWidth,
                int paddingHeight,
                int paddingWidth,
                int stride,
                int dilation)
{

    conv_fprop(toTensorRef(A_ptr, b, imageHeight, imageWidth, Din),
               toTensorRef(B_ptr, Dout, filterHeight, filterWidth, Din),
               toTensorRef(C_ptr,
                           b,
                           (imageHeight + 2 * paddingHeight - filterHeight) / stride + 1,
                           (imageWidth + 2 * paddingWidth - filterWidth) / stride + 1,
                           Dout),
               b,
               imageHeight,
               imageWidth,
               Din,
               Dout,
               filterHeight,
               filterWidth,
               paddingHeight,
               paddingWidth,
               stride,
               dilation);
}

}  // namespace gpu
