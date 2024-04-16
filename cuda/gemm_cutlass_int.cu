#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>


#include <iostream>
#include <gemm.cuh>

#include <cstdint>


template <typename Type>
void gemm_cutlass(int M, int N, int K, Type *X, Type *W, Type *Y) {
    Type *x, *w, *y;
    cudaMalloc((void **)&x, M * N * sizeof(Type));
    cudaMalloc((void **)&w, N * K * sizeof(Type));
    cudaMalloc((void **)&y, M * K * sizeof(Type));

    cudaMemcpy(x, X, M * N * sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(w, W, N * K * sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(y, Y, M * K * sizeof(Type), cudaMemcpyHostToDevice);

    gpu::gemm<Type>(M, N, K, x, false, w, false, y, false);

    cudaMemcpy(Y, y, M * K * sizeof(Type), cudaMemcpyDeviceToHost);

    cudaFree(x);
    cudaFree(w);
    cudaFree(y);
}

//forward declare the function for different integer Testtypes
template void gemm_cutlass<int8_t>(int M, int N, int K, int8_t *X, int8_t *W, int8_t *Y);
template void gemm_cutlass<int16_t>(int M, int N, int K, int16_t *X, int16_t *W, int16_t *Y);
template void gemm_cutlass<int32_t>(int M, int N, int K, int32_t *X, int32_t *W, int32_t *Y);
template void gemm_cutlass<int64_t>(int M, int N, int K, int64_t *X, int64_t *W, int64_t *Y);

