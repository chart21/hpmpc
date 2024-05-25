// Unfortunately HWCN to NHWC is not implemented
#pragma once
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/conv/kernel/default_conv2d_dgrad.h>
#include <cutlass/conv/kernel/default_conv2d_wgrad.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/device_chwn_to_nhwc.h>
#include <cutlass/util/device_nhwc_to_chwn.h>
#include <math.h>
#include <stdlib.h>
#include <thrust/device_vector.h>

#include <utils.cuh>


template<typename T>
inline cutlass::TensorRef<T, cutlass::layout::TensorNHWC> toTensorRef(
        T *ptr, int n, int h, int w, int c) {

    return cutlass::TensorRef<T, cutlass::layout::TensorNHWC>(
        ptr,
        cutlass::layout::TensorNHWC::packed({n, h, w, c})
    );
}


template<typename T>
inline cutlass::TensorRef<T, cutlass::layout::TensorNCHW> toTensorRefT(
        T *ptr, int n, int h, int w, int c) {

    return cutlass::TensorRef<T, cutlass::layout::TensorNCHW>(
        ptr,
        cutlass::layout::TensorNCHW::packed({n, c, h, w})
    );
}



template<typename T>
using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T, cutlass::layout::TensorNHWC,
    T,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
        T,
        1,
        T,
        T
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    4,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;


template<typename T>
using FpropImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel<T> >;


template<typename T>
struct FpropOptions {

    cutlass::Tensor4DCoord input_size;
    cutlass::Tensor4DCoord filter_size;
    cutlass::Tensor4DCoord padding;
    cutlass::MatrixCoord conv_stride;
    cutlass::MatrixCoord dilation;
    T alpha;
    T beta;

    FpropOptions(int in_n, int in_h, int in_w, int in_c,
            int f_k, int f_r, int f_s, int f_c, int padding_h, int padding_w,
            int _stride, int _dilation,
            T _alpha, T _beta) :
        input_size(in_n, in_h, in_w, in_c),
        filter_size(f_k, f_r, f_s, f_c),
        padding(padding_h, padding_h, padding_w, padding_w),
        conv_stride(_stride, _stride),
        dilation(_dilation, _dilation),
        alpha(_alpha), beta(_beta) { }

    cutlass::Tensor4DCoord output_size() const {
        return cutlass::Tensor4DCoord(
            input_size.n(),
            (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
            (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
            filter_size.n()
        );
    }
};


template<typename T>
cudaError_t CutlassConvFprop(
        const cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &A, 
        const cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &B, 
        cutlass::TensorRef<T, cutlass::layout::TensorNHWC> &C, 
        FpropOptions<T> const &options) {

    cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

    cutlass::conv::Conv2dProblemSize problem_size(
        options.input_size,
        options.filter_size,
        options.padding,
        options.conv_stride,
        options.dilation,
        options.output_size(),
        mode,
        1 // split_k_slices
    ); 

    typename FpropImplicitGemm<T>::Arguments arguments {
        problem_size,
        A, B, C, C,
        {options.alpha, options.beta} 
    };

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



namespace gpu{

template<typename T>
void conv_fprop(
        cutlass::TensorRef<T, cutlass::layout::TensorNCHW> AT,
        cutlass::TensorRef<T, cutlass::layout::TensorNCHW> BT,
        cutlass::TensorRef<T, cutlass::layout::TensorNCHW> CT,
        int b, int imageHeight, int imageWidth, int Din,
        int Dout, int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth,
        int stride, int dilation) {


    FpropOptions<T> options(
            b, imageHeight, imageWidth, Din,
            Dout, filterHeight, filterWidth, Din,
            paddingHeight, paddingWidth,
            stride, dilation, (T)1, (T)0);

    // transposing code starts
    cutlass::Tensor4DCoord ASize = options.input_size;     // nhwc
    cutlass::Tensor4DCoord BSize = options.filter_size;    // nhwc
    cutlass::Tensor4DCoord CSize = options.output_size();  // nhwc

    cutlass::Tensor4DCoord ASize_T = cutlass::Tensor4DCoord(ASize.n(), ASize.w(), ASize.c(), ASize.h());   // nchw
    cutlass::Tensor4DCoord BSize_T = cutlass::Tensor4DCoord(BSize.n(), BSize.w(), BSize.c(), BSize.h());   // nchw
    cutlass::Tensor4DCoord CSize_T = cutlass::Tensor4DCoord(CSize.n(), CSize.w(), CSize.c(), CSize.h());   // nchw

    T* Aptr;
    T* Bptr;
    T* Cptr;

    cudaMalloc((void **)&Aptr, b* imageHeight* imageWidth * Din * sizeof(T));
    cudaMalloc((void **)&Bptr, Dout * filterHeight * filterWidth * Din * sizeof(T));
    cudaMalloc((void **)&Cptr, CSize.n() * CSize.h() * CSize.w() * CSize.c() * sizeof(T));

    cutlass::TensorRef<T, cutlass::layout::TensorNHWC> A = toTensorRef(Aptr, b, imageHeight, imageWidth, Din);
    cutlass::TensorRef<T, cutlass::layout::TensorNHWC> B = toTensorRef(Bptr, Dout, filterHeight, filterWidth, Din);
    cutlass::TensorRef<T, cutlass::layout::TensorNHWC> C = toTensorRef(Cptr, CSize.n(), CSize.h(), CSize.w(), CSize.c());

    chwn_to_nhwc(ASize_T, ASize, AT, A, 0);
    chwn_to_nhwc(BSize_T, BSize, BT, B, 0);
    cudaDeviceSynchronize();
    // transposing code ends


    CutlassConvFprop<T>(A, B, C, options);
    cudaDeviceSynchronize();


    nhwc_to_chwn(CSize, CSize_T, C, CT, 0);
    cudaDeviceSynchronize();

    cudaFree(Aptr);
    cudaFree(Bptr);
    cudaFree(Cptr);
}


template<typename T>
void conv_fprop(T* A_ptr, T* B_ptr, T* C_ptr,
        int b, int imageHeight, int imageWidth, int Din,
        int Dout, int filterHeight, int filterWidth,
        int paddingHeight, int paddingWidth,
        int stride, int dilation) {


    conv_fprop(
        toTensorRefT(A_ptr, b, imageHeight, imageWidth, Din),
        toTensorRefT(B_ptr, Dout, filterHeight, filterWidth, Din),
        toTensorRefT(C_ptr, 
            b,
            (imageHeight + 2 * paddingHeight - filterHeight) / stride + 1,
            (imageWidth + 2 * paddingWidth - filterWidth) / stride + 1,
            Dout),
        b, imageHeight, imageWidth, Din, Dout, filterHeight, filterWidth,
        paddingHeight, paddingWidth, stride, dilation
    );
}


}  // namespace gpu
