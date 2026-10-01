#include <gemm.cuh>
#include "gemm_fast_gpu.h"

#include <cstdio>
#include <cstdlib>

namespace
{
void check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "gemm_fast_gpu: %s: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}
struct DevBuf
{
    uint32_t* p = nullptr;
    size_t n = 0;
    void need(size_t words)
    {
        if (n >= words)
            return;
        if (p)
            cudaFree(p);
        check(cudaMalloc(&p, words * sizeof(uint32_t)), "cudaMalloc");
        n = words;
    }
};
}  // namespace

uint32_t* gemm_fast_gpu_staging(size_t words, int which)
{
    static uint32_t* buf[2] = {nullptr, nullptr};
    static size_t cap[2] = {0, 0};
    which &= 1;
    if (cap[which] < words)
    {
        if (buf[which])
            cudaFreeHost(buf[which]);
        check(cudaMallocHost(&buf[which], words * sizeof(uint32_t)), "cudaMallocHost");
        cap[which] = words;
    }
    return buf[which];
}

void gemm_fast_gpu(int m, int p, int F, const uint32_t* X, bool x_fresh, const uint32_t* Bt, uint32_t* C)
{
    static DevBuf dx, db, dc;
    static cudaStream_t stream = [] {
        cudaStream_t s;
        check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        return s;
    }();
    dx.need(size_t(m) * F), db.need(size_t(F) * p), dc.need(size_t(m) * p);
    if (x_fresh)
        check(cudaMemcpyAsync(dx.p, X, size_t(m) * F * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), "copy X");
    check(cudaMemcpyAsync(db.p, Bt, size_t(F) * p * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), "copy B");
    check(cudaStreamSynchronize(stream), "copies");
    gpu::gemm<uint32_t>(m, p, F, dx.p, true, db.p, true, dc.p, true);  // row-major operands and result
    check(cudaDeviceSynchronize(), "gemm");
    check(cudaMemcpyAsync(C, dc.p, size_t(m) * p * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "copy C");
    check(cudaStreamSynchronize(stream), "copy C");
}
