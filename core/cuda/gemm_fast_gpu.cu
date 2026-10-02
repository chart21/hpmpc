#include <gemm.cuh>
#include "gemm_fast_gpu.h"

#include <algorithm>
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

namespace
{
// row k = (c, dy, dx) of the column matrix, column n = (pixel, lane): Y[c][iy][ix][lane] or 0 outside the image
__global__ void im2col_lanes(uint32_t* out, const uint32_t* __restrict__ Y, int ic, int ih, int iw, int ksz, int stride,
                             int pad, int ow, int p, int L, long long total)
{
    for (long long t = (long long) blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (long long) gridDim.x * blockDim.x)
    {
        const long long pl = (long long) p * L;
        const int k = int(t / pl);
        const long long n = t % pl;
        const int pix = int(n / L), lane = int(n % L);
        const int kk = ksz * ksz, c = k / kk, r = k % kk, dy = r / ksz, dx = r % ksz;
        const int iy = (pix / ow) * stride - pad + dy, ix = (pix % ow) * stride - pad + dx;
        out[t] = (iy >= 0 && iy < ih && ix >= 0 && ix < iw) ? Y[(((long long) c * ih + iy) * iw + ix) * L + lane] : 0u;
    }
}
}  // namespace

void conv_fast_gpu(int m, int ic, int ih, int iw, int ksz, int stride, int pad, int L, const uint32_t* X, bool x_fresh,
                   const uint32_t* Y, uint32_t* C)
{
    static DevBuf dx, dy, db, dc;
    static cudaStream_t stream = [] {
        cudaStream_t s;
        check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        return s;
    }();
    const int oh = (ih + 2 * pad - ksz) / stride + 1, ow = (iw + 2 * pad - ksz) / stride + 1, p = oh * ow;
    const int F = ic * ksz * ksz;
    const size_t pl = size_t(p) * L, ny = size_t(ic) * ih * iw * L;
    dx.need(size_t(m) * F), dy.need(ny), db.need(size_t(F) * pl), dc.need(size_t(m) * pl);
    if (x_fresh)
        check(cudaMemcpyAsync(dx.p, X, size_t(m) * F * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), "copy X");
    check(cudaMemcpyAsync(dy.p, Y, ny * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), "copy Y");
    const long long total = (long long) F * (long long) pl;
    const unsigned blocks = unsigned(std::min<long long>((total + 255) / 256, 65535LL * 8));
    im2col_lanes<<<blocks, 256, 0, stream>>>(db.p, dy.p, ic, ih, iw, ksz, stride, pad, ow, p, L, total);
    check(cudaGetLastError(), "im2col");
    check(cudaStreamSynchronize(stream), "im2col");
    gpu::gemm<uint32_t>(m, int(pl), F, dx.p, true, db.p, true, dc.p, true);  // row-major operands and result
    check(cudaDeviceSynchronize(), "gemm");
    check(cudaMemcpyAsync(C, dc.p, size_t(m) * pl * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), "copy C");
    check(cudaStreamSynchronize(stream), "copy C");
}
