#pragma once
// The product of programs/functions/GEMM_fast.hpp on the GPU (GEMM_FAST_GPU=1; link core/cuda/bin/gemm_fast_gpu.o and
// cudart): C = X Bt over uint32 (ring arithmetic mod 2^32), X m x F and Bt F x p row-major. The weight operand X stays on
// the device between calls of a layer (x_fresh: a new one).
#include <cstddef>
#include <cstdint>

// pinned host memory of at least `words` words (which: 0 the input operand, 1 the result), kept across calls
uint32_t* gemm_fast_gpu_staging(size_t words, int which);
void gemm_fast_gpu(int m, int p, int F, const uint32_t* X, bool x_fresh, const uint32_t* Bt, uint32_t* C);
