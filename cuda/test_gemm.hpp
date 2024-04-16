#pragma once
#include "gemm_cutlass_int.h"
#include <iostream>
#include <cstdint>

void test_cuda() {
  int M = 2;
  int N = 2;
  int K = 2;
  int16_t X[2 * 2] = {1, 2, 3, 4};
  int16_t W[2 * 2] = {1, 2, 3, 4};
  int16_t Y[2 * 2] = {0, 0, 0, 0};
  gemm_cutlass(M, N, K, X, W, Y);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << Y[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}
