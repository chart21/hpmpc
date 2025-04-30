#pragma once
#include <cstdint>

template <typename Type>
void gemm_cutlass(int M, int N, int K, Type* X, Type* W, Type* Y);
