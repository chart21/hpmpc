#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>


#include <iostream>
#include <gemm.cuh>

#include <cstdint>


template <typename Type>
void gemm_cutlass(int M, int N, int K, Type *X, Type *W, Type *Y) {
    Type *x, *w, *y;
    cudaMalloc((void **)&x, M * K * sizeof(Type)); //Matrix X has M rows and K columns
    cudaMalloc((void **)&w, K * N * sizeof(Type)); //Matrix W has K rows and N columns
    cudaMalloc((void **)&y, M * N * sizeof(Type)); //Matrix Y has M rows and N columns

    cudaMemcpy(x, X, M * K * sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(w, W, K * N * sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(y, Y, M * N * sizeof(Type), cudaMemcpyHostToDevice);

    gpu::gemm<Type>(M, N, K, x, true, w, true, y, true); //true means that the matrix is in row-major order

    cudaMemcpy(Y, y, M * N * sizeof(Type), cudaMemcpyDeviceToHost);

    cudaFree(x);
    cudaFree(w);
    cudaFree(y);
}

//forward declare the function for different integer Testtypes
template void gemm_cutlass<int8_t>(int M, int N, int K, int8_t *X, int8_t *W, int8_t *Y);
template void gemm_cutlass<int16_t>(int M, int N, int K, int16_t *X, int16_t *W, int16_t *Y);
template void gemm_cutlass<int32_t>(int M, int N, int K, int32_t *X, int32_t *W, int32_t *Y);
template void gemm_cutlass<int64_t>(int M, int N, int K, int64_t *X, int64_t *W, int64_t *Y);
template void gemm_cutlass<unsigned int>(int M, int N, int K, unsigned int *X, unsigned int *W, unsigned int *Y);
