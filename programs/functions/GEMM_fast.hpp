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
#include <atomic>
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
// Multi-batch (DATTYPE = L * BITLENGTH, L images in the lanes of a register): every lane holds the same weights, so a
// product is one GEMM with L times the columns (column j, lane l); the weight operand is taken from lane 0 after
// checking that all lanes agree (else the share-level loop runs).
#define GEMM_FAST_ELIGIBLE (GEMM_FAST == 1 && A_KNOWN == 1 && PROTOCOL == 4 && DATTYPE % BITLENGTH == 0 && PUBLIC_WEIGHTS == 0 && FUSE_CONV_BN_SIM == 0 && \
                            ADDITIONAL_GEMM_THREADS > 0)

#if GEMM_FAST_ELIGIBLE
namespace gemm_fast
{
using W = UINT_TYPE;
constexpr int L = DATTYPE / BITLENGTH;  // lanes (images) per register
constexpr int NR = 32;  // columns per panel (two AVX-512 / four AVX2 vectors)
constexpr int MR = 4;   // rows per block
constexpr int K = (A_KNOWN == 0) ? 2 : 1;  // A_KNOWN == 0: two products, concatenated along k
static_assert(NR % L == 0, "GEMM_FAST: a panel holds whole registers");

// the L ring elements of a share word
template <typename D>
inline const W* lanes(const D& v)
{
    return reinterpret_cast<const W*>(&v);
}
template <typename D>
inline W* lanes(D& v)
{
    return reinterpret_cast<W*>(&v);
}

// the weight-side operand x of one row (length K * f), from lane 0; false if another lane differs
template <typename U>
inline bool x_row(const U* a, int f, W* x)
{
    bool same = true;
    for (int k = 0; k < f; k++)
    {
        const W* m = lanes(a[k].raw_m());
        const W* l = lanes(a[k].raw_l());
        for (int q = 0; q < L; q++)
        {
#if A_KNOWN == 0
            const W v0 = m[q], v1 = l[q];
#elif MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
            const W v0 = l[q], v1 = 0;
#elif PARTY == 0
            const W v0 = m[q] - l[q], v1 = 0;
#else
            const W v0 = m[q], v1 = 0;
#endif
            if (q == 0)
            {
                x[k] = v0;
                if constexpr (K == 2)
                    x[f + k] = v1;
            }
            else
                same &= v0 == x[k] && (K == 1 || v1 == x[(K - 1) * f + k]);
        }
    }
    return same;
}

// the input-side operand y of one column's L lanes, written to columns 0 .. L - 1 of rows of stride `stride` (a panel:
// NR; the row-major operand of the GPU product: p * L)
template <typename T>
inline void y_col(const T* b, int f, W* y, size_t stride = NR)
{
    for (int k = 0; k < f; k++)
    {
        const W* m = lanes(b[k].raw_m());
        const W* l = lanes(b[k].raw_l());
        W* yk = y + (size_t) k * stride;
        for (int q = 0; q < L; q++)
        {
#if A_KNOWN == 0
#if PARTY == 0
            yk[q] = m[q] - l[q];
#else
            yk[q] = W(0) - l[q];
#endif
            y[(size_t) (f + k) * stride + q] = W(0) - m[q];
#elif PARTY == 0
            yk[q] = m[q] - l[q];
#else
            yk[q] = W(0) - l[q];
#endif
        }
    }
}

// C[i0 + r][j0 + v / L] lane v % L .m += sum_k x_r[k] * yp[k][v] for r < R, v < ncols
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
            lanes(c[r][v / L].raw_m())[v % L] += acc[r][v];
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
        static bool x_same = true;  // every lane has lane 0's weights
        const bool fresh = x_of != (const void*) A || x_m != m || x_f != f;
        if (fresh)
        {
            X.resize((size_t) m * F);
            std::atomic<bool> same{true};
            GemmPool::get().run([&](int t) {
                bool s = true;
                for (int i = m * t / TT; i < m * (t + 1) / TT; i++)
                    s &= x_row(A + (size_t) i * f, f, X.data() + (size_t) i * F);
                if (!s)
                    same = false;
            });
            x_of = A, x_m = m, x_f = f, x_same = same;
        }
        if (!x_same)
            return false;  // lanes with different weights: the share-level loop
        const int pl = p * L;  // columns of the product: (column, lane)
#if GEMM_FAST_GPU
        {
            // the product on the GPU (core/cuda/gemm_fast_gpu.cu): the input operand row-major, F x p, then C.m += the result
            static_assert(sizeof(W) == sizeof(uint32_t), "GEMM_FAST_GPU: 32-bit ring only");
            W* bt = gemm_fast_gpu_staging((size_t) F * pl, 0);
            W* res = gemm_fast_gpu_staging((size_t) m * pl, 1);
            GemmPool::get().run([&](int t) {
                for (int j = p * t / TT; j < p * (t + 1) / TT; j++)
                    y_col(B + (size_t) j * f, f, bt + (size_t) j * L, (size_t) pl);
            });
            gemm_fast_gpu(m, pl, F, X.data(), fresh, bt, res);
            static const bool verify = getenv("GEMM_FAST_GPU_CHECK") && atoi(getenv("GEMM_FAST_GPU_CHECK")) != 0;
            if (verify)  // 64 sampled entries of every product, recomputed here
                for (int s = 0; s < 64; s++)
                {
                    const size_t i = ((size_t) s * 2654435761u) % m, j = ((size_t) s * 40503u + 7) % pl;
                    W ref = 0;
                    for (int k = 0; k < F; k++)
                        ref += X[i * F + k] * bt[(size_t) k * pl + j];
                    if (ref != res[i * pl + j])
                    {
                        fprintf(stderr, "GEMM_FAST_GPU: C[%zu][%zu] of %d x %d x %d is %u, not %u\n", i, j, m, pl, F,
                                (unsigned) res[i * pl + j], (unsigned) ref);
                        std::abort();
                    }
                }
            GemmPool::get().run([&](int t) {
                for (size_t e = (size_t) m * p * t / TT; e < (size_t) m * p * (t + 1) / TT; e++)
                {
                    W* c = lanes(C[e].raw_m());
                    const W* r = res + e * L;
                    for (int q = 0; q < L; q++)
                        c[q] += r[q];
                }
            });
            return true;
        }
#endif
        const int npan = (pl + NR - 1) / NR;
        yp.resize((size_t) npan * F * NR);
        GemmPool::get().run([&](int t) {
            for (int pn = npan * t / TT; pn < npan * (t + 1) / TT; pn++)
            {
                W* dst = yp.data() + (size_t) pn * F * NR;
                const int v0 = pn * NR, nc = std::min(NR, pl - v0);
                for (int v = 0; v < nc; v += L)
                    y_col(B + (size_t) ((v0 + v) / L) * f, f, dst + v);
                for (int v = nc; v < NR; v++)  // padding columns of the last panel
                    for (int k = 0; k < F; k++)
                        dst[(size_t) k * NR + v] = W(0);
            }
        });
        const int nrb = (m + MR - 1) / MR;
        const long long blocks = (long long) npan * nrb;  // panel-major: consecutive blocks reuse a panel
        GemmPool::get().run([&](int t) {
            for (long long b = blocks * t / TT; b < blocks * (t + 1) / TT; b++)
            {
                const int pn = int(b / nrb), rb = int(b % nrb);
                const int i0 = rb * MR, j0 = pn * NR / L, nc = std::min(NR, pl - pn * NR), nr = std::min(MR, m - i0);
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
