#pragma once
#include "../../config.h"
#include "prob_truncation.hpp"
#include <algorithm>

template <typename T>
void prepare_Matrix_Vector_Product(const T* W, const T* A, T* C, const int w_rows, const int w_cols)
{
    for (int i = 0; i < w_rows; ++i)
    {
        T sum = T(0);
        for (int j = 0; j < w_cols; ++j)
        {

#if PUBLIC_WEIGHTS == 0
            sum += W[i * w_cols + j].prepare_dot(A[j]);
#else
            sum += A[j].mult_public(W[i * w_cols + j]);
#endif
        }

#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
        sum.mask_and_send_dot_without_trunc();  // send immediately to utilize network better
#else
        sum.mask_and_send_dot();
#endif
#else
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
        // do nothing
#else
        sum = sum.prepare_mult_public_fixed(1);  // initiate truncation
#endif
#endif

        C[i] = sum;
    }
}

template <typename T>
void prepare_GEMM_CPU(const T* A, const T* B, T* C, const int m, const int p, const int f, bool is_A_fixed)
{
    const int TILE_SIZE = 64;

#if FUSE_DOT == 2
    T* Q = new T[m * p * T::getNumDotProducts()];
    for (int t = 0; t < T::getNumDotProducts(); t++)
    {
#endif
        for (int i = 0; i < m; i += TILE_SIZE)
        {
            int i_max = std::min(i + TILE_SIZE, m);
            for (int j = 0; j < p; j += TILE_SIZE)
            {
                /* _mm_prefetch(B + j * f, _MM_HINT_T0); */
                int j_max = std::min(j + TILE_SIZE, p);
                for (int k = 0; k < f; k += TILE_SIZE)
                {
                    int k_max = std::min(k + TILE_SIZE, f);
                    for (int ii = i; ii < i_max; ++ii)
                    {
                        const int iip = ii * p;
                        const int iif = ii * f;
                        /* const int row2 = ii*f+kk; */
                        for (int jj = j; jj < j_max; ++jj)
                        {
                            const int jjf = jj * f;
                            auto temp = T(0);
#if FUSE_DOT == 0
                            T temps[T::getNumDotProducts()] = {T(0)};
#endif
                            for (int kk = k; kk < k_max; ++kk)
                            {
                                /* _mm_prefetch(C + ii * p + jj, _MM_HINT_T0); */
#if PUBLIC_WEIGHTS == 0
#if FUSE_DOT == 0
                                for (int i = 0; i < T::getNumDotProducts(); ++i)
                                {
                                    temps[i] += A[iif + kk].prepare_dot(B[jjf + kk], i);
                                }
                                temp.join_dots(temps);
#elif FUSE_DOT == 1
                                temp += A[iif + kk].prepare_dot(B[jjf + kk]);
#elif FUSE_DOT == 2
                                Q[iip + jj + t * m * p] += A[iif + kk].prepare_dot(B[jjf + kk], t);
#endif
#else

                            temp += B[jjf + kk].mult_public(A[iif + kk]);
#endif
                            }
#if FUSE_DOT == 2
                            Q[iip + jj + t * m * p] = temp;
#else
                        C[iip + jj] += temp;
#endif
                        }
                    }
                }
#if INTERLEAVE_COMM == 1
                for (int ii = i; ii < i_max; ++ii)
                {
                    const int row = ii * p;
                    for (int jj = j; jj < j_max; ++jj)
                    {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
                        C[row + jj].mask_and_send_dot_without_trunc();
#else
                        C[row + jj].mask_and_send_dot();
#endif
#else
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
                        // do nothing
#else
                        C[row + jj] = C[row + jj].prepare_mult_public_fixed(1);  // initiate truncation
#endif
#endif
                    }
                }
#endif
            }
        }
#if FUSE_DOT == 2
    }
#endif
#if FUSE_DOT == 2
    for (int i = 0; i < m * p; ++i)
    {
        T temps[T::getNumDotProducts()];
        for (int t = 0; t < T::getNumDotProducts(); t++)
        {
            temps[t] = Q[i + t * m * p];
        }
        C[i].join_dots(temps);
        C[i].mask_and_send_dot();
    }
    delete[] Q;
#else
#if INTERLEAVE_COMM == 0
    for (int i = 0; i < m * p; ++i)
        C[i].mask_and_send_dot();
#endif
#endif
}

template <typename T>
void complete_GEMM_CPU(T* C, const int m, const int p)
{
    const int TILE_SIZE = 64;
    for (int i = 0; i < m; i += TILE_SIZE)
    {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE)
        {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int ii = i; ii < i_max; ++ii)
            {
                const int row = ii * p;
                for (int jj = j; jj < j_max; ++jj)
                {
                    /* C[row + jj].complete_mult(); */
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
                    C[row + jj].complete_mult_without_trunc();
#else
                    C[row + jj].complete_mult();
#endif
#else
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
#else
                    C[row + jj].complete_mult_public_fixed();
#endif
#endif
                }
            }
        }
    }
}

template <typename T>
void send_GEMM_GPU(T* C, const int m, const int p)
{
    for (int j = 0; j < m * p; ++j)
    {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
        C[j].mask_and_send_dot_without_trunc();
#else
        C[j].mask_and_send_dot();
#endif
#else
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
        // do nothing
#else
        C[j] = C[j].prepare_mult_public_fixed(1);  // initiate truncation
#endif
#endif
    }
}

template <typename T>
void complete_GEMM(T* C, const int len)
{
    for (int i = 0; i < len; ++i)
    {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
        C[i].complete_mult_without_trunc();
#else
        C[i].complete_mult();
#endif
#else
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
#else
        C[i].complete_mult_public_fixed();
#endif
#endif
    }
}

template <typename T>
void prepare_GEMM(T* A, T* B, T* C, const int m, const int p, const int f, bool is_A_fixed)
{
#if USE_CUDA_GEMM == 0
    prepare_GEMM_CPU(A, B, C, m, p, f, is_A_fixed);
#else
    T::GEMM(A, B, C, m, p, f, is_A_fixed);
    send_GEMM_GPU(C, m, p);
#endif
}
/* T::CONV_2D( im,W,C, local_batch, ih, iw, ic, oc, kh, kw, pad, stride, 1); */

template <typename T>
void complete_GEMM(T* C, const int m, const int p)
{
#if USE_CUDA_GEMM == 0
    complete_GEMM_CPU(C, m, p);
#else
    complete_GEMM(C, m * p);
#endif
}

template <typename T>
void add_bias(T& C, const T& bias)
{
#if TRUNC_DELAYED == 0
    C += bias;
#else
    // multiply each bias by 2^FRACTIONAL
#if PUBLIC_WEIGHTS == 0
    C += bias.mult_public(UINT_TYPE(1) << FRACTIONAL);
#else
    C += bias << FRACTIONAL;
#endif
#endif
}
