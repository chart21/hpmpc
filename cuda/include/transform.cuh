#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"
#include <cutlass/util/device_nchw_to_nhwc.h>
#include <cutlass/util/device_nhwc_to_nchw.h>


template <typename T>
void nchw_to_nhwc_(
    T *output, 
    const T *input, 
    const int n,
    const int h, 
    const int w, 
    const int c) { 
    
    dim3 grid((h*w + 31)/32, (c + 31)/32, n);
    dim3 block(32, 8);
    cutlass::nchw_to_nhwc_kernel<<<grid, block, 0,  0>>>(output, input, n,h,w,c);
}





template <typename T>
void chwn_to_nhwc_(
    T *output, 
    const T *input, 
    const int n,
    const int h, 
    const int w, 
    const int c) { 
    
    int newN = 1;
    int newC = c*h*w;
    int newH = 1;
    int newW = n;

    T* buf;
    int size = n*c*h*w;
    cudaMalloc((void**) &buf, sizeof(T)*size);

    //nchw_to_nhwc_(buf, input, newN, newH, newW, newC);
    nchw_to_nhwc_(output, buf, n, h, w, c);

    cudaFree(buf);
}


template <typename T>
void chwn_to_nhwc(cutlass::Tensor4DCoord input_tensor_size,
                  cutlass::Tensor4DCoord output_tensor_size,
                  cutlass::TensorRef<T, cutlass::layout::TensorNCHW> ref_input,
                  cutlass::TensorRef<T, cutlass::layout::TensorNHWC> ref_output) {
  
  assert(
    input_tensor_size.c() == output_tensor_size.n() &&
    input_tensor_size.h() == output_tensor_size.h() &&
    input_tensor_size.w() == output_tensor_size.w() &&
    input_tensor_size.n() == output_tensor_size.c());

  int n = output_tensor_size.n();
  int h = output_tensor_size.h();
  int w = output_tensor_size.w();
  int c = output_tensor_size.c();
  
  chwn_to_nhwc_(ref_output.data(), ref_input.data(), n, h, w, c);
}


template <typename T>
void nhwc_to_nchw_(
    T *output, 
    const T *input, 
    const int n,
    const int h, 
    const int w, 
    const int c) { 
    
    dim3 grid((c + 31)/32, (h*w + 31)/32, n);
    dim3 block(32, 8);
    cutlass::nhwc_to_nchw_kernel<<<grid, block, 0,  0>>>(output, input, n,h,w,c);
}


template <typename T>
void nhwc_to_chwn_(
    T *output, 
    const T *input, 
    const int n,
    const int h, 
    const int w, 
    const int c) { 
    
    int newN = 1;
    int newC = c*h*w;
    int newH = 1;
    int newW = n;

    T* buf;
    int size = n*c*h*w;
    cudaMalloc((void**) &buf, sizeof(T)*size);

    nhwc_to_nchw_(buf, input, n, h, w, c);
    nhwc_to_nchw_(output, buf, newN, newH, newW, newC);

    cudaFree(buf);
}


template <typename T>
void nhwc_to_chwn(cutlass::Tensor4DCoord input_tensor_size,
                  cutlass::Tensor4DCoord output_tensor_size,
                  cutlass::TensorRef<T, cutlass::layout::TensorNCHW> ref_input,
                  cutlass::TensorRef<T, cutlass::layout::TensorNHWC> ref_output) {
  
  assert(
    input_tensor_size.n() == output_tensor_size.c() &&
    input_tensor_size.h() == output_tensor_size.h() &&
    input_tensor_size.w() == output_tensor_size.w() &&
    input_tensor_size.c() == output_tensor_size.n());

  int n = input_tensor_size.n();
  int h = input_tensor_size.h();
  int w = input_tensor_size.w();
  int c = input_tensor_size.c();
  
  nhwc_to_chwn_(ref_output.data(), ref_input.data(), n, h, w, c);
}
