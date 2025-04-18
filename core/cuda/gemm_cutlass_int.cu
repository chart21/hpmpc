#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>
#include <gemm.cuh>
#include <cstdint>
#include <iostream>

template <typename Type>
void gemm_cutlass(int M, int N, int K, Type* X, Type* W, Type* Y)
{
    Type *x, *w, *y;
    cudaMalloc((void**)&x, M * K * sizeof(Type));  // Matrix X has M rows and K columns
    cudaMalloc((void**)&w, K * N * sizeof(Type));  // Matrix W has K rows and N columns
    cudaMalloc((void**)&y, M * N * sizeof(Type));  // Matrix Y has M rows and N columns

    cudaMemcpy(x, X, M * K * sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(w, W, K * N * sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(y, Y, M * N * sizeof(Type), cudaMemcpyHostToDevice);

    gpu::gemm<Type>(M, N, K, x, true, w, true, y, true);  // true means that the matrix is in row-major order

    cudaMemcpy(Y, y, M * N * sizeof(Type), cudaMemcpyDeviceToHost);

    cudaFree(x);
    cudaFree(w);
    cudaFree(y);
}

// forward declare the function for different integer Testtypes
//  UINT8 and UINT16 are not supported by all architectures
// template void gemm_cutlass<uint8_t>(int M, int N, int K, uint8_t *X, uint8_t *W, uint8_t *Y);
template void gemm_cutlass<uint16_t>(int M,
                                     int N,
                                     int K,
                                     uint16_t* X,
                                     uint16_t* W,
                                     uint16_t* Y);  // INT8 and INT16 are not supported by some architectures
template void gemm_cutlass<uint32_t>(int M, int N, int K, uint32_t* X, uint32_t* W, uint32_t* Y);
template void gemm_cutlass<uint64_t>(int M, int N, int K, uint64_t* X, uint64_t* W, uint64_t* Y);
