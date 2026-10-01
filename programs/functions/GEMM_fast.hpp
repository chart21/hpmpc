#pragma once
// Vectorized accumulation of the threaded GEMM (GEMM_threaded.hpp) for ABY2 with secret weights, single batch
// (DATTYPE == BITLENGTH: one ring element per share word) with the weight masks known to P0 (A_KNOWN == 1; with
// A_KNOWN == 0 the two concatenated products were not faster than the share-level loop). The share-level loop called a prepare_dot kernel per
// multiply-accumulate on {m, l} structs and did not vectorize (~0.2 MAC per cycle). Each kernel is a sum of plain
// ring products, so it is extracted into uint32 matrices and computed as C = X * Y^T with a register-blocked kernel
// (4 rows x 32 columns). Ring arithmetic is exact, so every output is bit-identical to the share-level loop.
//   MODELWEIGHTS_KNOWN (a_known_pre): P0  C.m += A.l (B.m - B.l)             P1  nothing (its mask/send ignores C)
//   A_KNOWN == 1 (a_known):           P0  C.m += (A.m - A.l)(B.m - B.l)      P1  C.m += A.m (-B.l)
//   A_KNOWN == 0 (ex_lxly):           P0  C.m += A.m (B.m - B.l) + A.l (-B.m)  P1  C.m += A.m (-B.l) + A.l (-B.m)
// The share-level kernels also accumulated the never-initialized l of their result into C.l; every mask/send
// overwrites C.l, so the fast path leaves it alone. In the preprocessing pass these kernels return zero shares:
// the accumulation is skipped there.
#include "../../config.h"
#include "worker_pool.hpp"
#include <cstdint>
#include <type_traits>
#include <vector>

#ifndef GEMM_FAST
#define GEMM_FAST 1  // 0: the share-level accumulation (A/B checks)
#endif
#ifndef GEMM_FAST_GPU
#define GEMM_FAST_GPU 0  // 1: GEMM_FAST's product on the GPU (link core/cuda/bin/gemm_fast_gpu.o and cudart)
#endif
#if GEMM_FAST_GPU
#include "../../core/cuda/gemm_fast_gpu.h"
#endif
#define GEMM_FAST_ELIGIBLE (GEMM_FAST == 1 && A_KNOWN == 1 && PROTOCOL == 4 && DATTYPE == BITLENGTH && PUBLIC_WEIGHTS == 0 && FUSE_CONV_BN_SIM == 0 && \
                            ADDITIONAL_GEMM_THREADS > 0)

#if GEMM_FAST_ELIGIBLE
namespace gemm_fast
{
using W = UINT_TYPE;
constexpr int NR = 32;  // columns per panel (two AVX-512 / four AVX2 vectors)
constexpr int MR = 4;   // rows per block
constexpr int K = (A_KNOWN == 0) ? 2 : 1;  // A_KNOWN == 0: two products, concatenated along k

// the weight-side operand x of one row (length K * f)
template <typename U>
inline void x_row(const U* a, int f, W* x)
{
    for (int k = 0; k < f; k++)
    {
#if A_KNOWN == 0
        x[k] = a[k].raw_m();
        x[f + k] = a[k].raw_l();
#elif MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
        x[k] = a[k].raw_l();
#elif PARTY == 0
        x[k] = a[k].raw_m() - a[k].raw_l();
#else
        x[k] = a[k].raw_m();
#endif
    }
}

// the input-side operand y of one column, written with stride NR into a panel (length K * f), or with stride p into the
// row-major operand of the GPU product
template <typename T>
inline void y_col(const T* b, int f, W* y, size_t stride = NR)
{
    for (int k = 0; k < f; k++)
    {
#if A_KNOWN == 0
#if PARTY == 0
        y[(size_t) k * stride] = b[k].raw_m() - b[k].raw_l();
#else
        y[(size_t) k * stride] = W(0) - b[k].raw_l();
#endif
        y[(size_t) (f + k) * stride] = W(0) - b[k].raw_m();
#elif PARTY == 0
        y[(size_t) k * stride] = b[k].raw_m() - b[k].raw_l();
#else
        y[(size_t) k * stride] = W(0) - b[k].raw_l();
#endif
    }
}

// C[i0 + r][j0 + v].m += sum_k x_r[k] * yp[k][v] for r < R, v < ncols
template <int R, typename T>
inline void kernel(const W* const* x, const W* yp, int F, T* const* c, int ncols)
{
    W acc[R][NR] = {};
    for (int k = 0; k < F; k++)
    {
        const W* y = yp + (size_t) k * NR;
        W a[R];
        for (int r = 0; r < R; r++)
            a[r] = x[r][k];
        for (int r = 0; r < R; r++)
#pragma GCC ivdep
            for (int v = 0; v < NR; v++)
                acc[r][v] += a[r] * y[v];
    }
    for (int r = 0; r < R; r++)
        for (int v = 0; v < ncols; v++)
            c[r][v].raw_m() += acc[r][v];
}

// true: the accumulation was done (or is not needed) here
template <typename T, typename U>
inline bool accumulate(const U* A, const T* B, T* C, int m, int p, int f)
{
    if constexpr (!std::is_same_v<T, U> || !requires(T& t) { t.raw_m(); })
    {
        // the preprocessing pass: every secret-weight dot kernel returns a zero share
        return current_phase == PHASE_PRE;
    }
    else
    {
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1 && A_KNOWN == 1 && PARTY == 1
        return true;
#endif
        const int F = K * f;
        constexpr int TT = WORKER_POOL_THREADS + 1;
        // x depends only on the weights: a layer's per-image calls (same A) extract it once
        static std::vector<W> X, yp;
        static const void* x_of = nullptr;
        static int x_m = 0, x_f = 0;
        const bool fresh = x_of != (const void*) A || x_m != m || x_f != f;
        if (fresh)
            X.resize((size_t) m * F);
#if GEMM_FAST_GPU
        {
            // the product on the GPU (core/cuda/gemm_fast_gpu.cu): the input operand row-major, F x p, then C.m += the result
            static_assert(sizeof(W) == sizeof(uint32_t), "GEMM_FAST_GPU: 32-bit ring only");
            W* bt = gemm_fast_gpu_staging((size_t) F * p, 0);
            W* res = gemm_fast_gpu_staging((size_t) m * p, 1);
            GemmPool::get().run([&](int t) {
                if (fresh)
                    for (int i = m * t / TT; i < m * (t + 1) / TT; i++)
                        x_row(A + (size_t) i * f, f, X.data() + (size_t) i * F);
                for (int j = p * t / TT; j < p * (t + 1) / TT; j++)
                    y_col(B + (size_t) j * f, f, bt + j, (size_t) p);
            });
            x_of = A, x_m = m, x_f = f;
            gemm_fast_gpu(m, p, F, X.data(), fresh, bt, res);
            static const bool verify = getenv("GEMM_FAST_GPU_CHECK") && atoi(getenv("GEMM_FAST_GPU_CHECK")) != 0;
            if (verify)  // 64 sampled entries of every product, recomputed here
                for (int s = 0; s < 64; s++)
                {
                    const size_t i = ((size_t) s * 2654435761u) % m, j = ((size_t) s * 40503u + 7) % p;
                    W ref = 0;
                    for (int k = 0; k < F; k++)
                        ref += X[i * F + k] * bt[(size_t) k * p + j];
                    if (ref != res[i * p + j])
                    {
                        fprintf(stderr, "GEMM_FAST_GPU: C[%zu][%zu] of %d x %d x %d is %u, not %u\n", i, j, m, p, F,
                                (unsigned) res[i * p + j], (unsigned) ref);
                        std::abort();
                    }
                }
            GemmPool::get().run([&](int t) {
                for (size_t e = (size_t) m * p * t / TT; e < (size_t) m * p * (t + 1) / TT; e++)
                    C[e].raw_m() += res[e];
            });
            return true;
        }
#endif
        const int npan = (p + NR - 1) / NR;
        yp.resize((size_t) npan * F * NR);
        GemmPool::get().run([&](int t) {
            if (fresh)
                for (int i = m * t / TT; i < m * (t + 1) / TT; i++)
                    x_row(A + (size_t) i * f, f, X.data() + (size_t) i * F);
            for (int pn = npan * t / TT; pn < npan * (t + 1) / TT; pn++)
            {
                W* dst = yp.data() + (size_t) pn * F * NR;
                const int j0 = pn * NR, nc = std::min(NR, p - j0);
                for (int v = 0; v < nc; v++)
                    y_col(B + (size_t) (j0 + v) * f, f, dst + v);
                for (int v = nc; v < NR; v++)  // padding columns of the last panel
                    for (int k = 0; k < F; k++)
                        dst[(size_t) k * NR + v] = W(0);
            }
        });
        x_of = A, x_m = m, x_f = f;
        const int nrb = (m + MR - 1) / MR;
        const long long blocks = (long long) npan * nrb;  // panel-major: consecutive blocks reuse a panel
        GemmPool::get().run([&](int t) {
            for (long long b = blocks * t / TT; b < blocks * (t + 1) / TT; b++)
            {
                const int pn = int(b / nrb), rb = int(b % nrb);
                const int i0 = rb * MR, j0 = pn * NR, nc = std::min(NR, p - j0), nr = std::min(MR, m - i0);
                const W* ypn = yp.data() + (size_t) pn * F * NR;
                const W* x[MR];
                T* c[MR];
                for (int r = 0; r < nr; r++)
                    x[r] = X.data() + (size_t) (i0 + r) * F, c[r] = C + (size_t) (i0 + r) * p + j0;
                if (nr == MR)
                    kernel<MR>(x, ypn, F, c, nc);
                else
                    for (int r = 0; r < nr; r++)
                        kernel<1>(x + r, ypn, F, c + r, nc);
            }
        });
        return true;
    }
}
}  // namespace gemm_fast
#endif
