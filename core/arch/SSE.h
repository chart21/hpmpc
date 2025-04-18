/* ******************************************** *\
 *
 *
 *
\* ******************************************** */

/* Including headers */
#pragma once
#include "../../config.h"
#include "../include/pch.h"
#include <wmmintrin.h>
#include <x86intrin.h>
#ifndef SSE
#define SSE_DAT
#endif

#ifndef BITS_PER_REG
#define BITS_PER_REG 128
#endif

/* #ifdef __WMMINTRIN_AES_H */
/* #define MM_XOR _mm_xor_si128 */
/* #define MM_AES_ENC _mm_aesenc_si128 */
/* #define MM_AES_ENC_LAST _mm_aesenclast_si128 */
/* #define MM_AES_DEC _mm_aesdec_si128 */
/* #define MM_AES_DEC_LAST _mm_aesdeclast_si128 */
/* #define MM_AES_KEYGEN _mm_aeskeygenassist_si128 */
/* #define MM_AES_KEYGEN_LAST _mm_aeskeygenassist_si128 */
/* #endif */

/* Defining 0 and 1 */
#define ZERO _mm_setzero_si128()
#define ONES _mm_set1_epi32(-1)
#if BITLENGTH == 16
#define PROMOTE(x) _mm_set1_epi16(x)
#elif BITLENGTH == 32
#define PROMOTE(x) _mm_set1_epi32(x)
#elif BITLENGTH == 64
#define PROMOTE(x) _mm_set1_epi64x(x)
#endif

/* Defining macros */
#define REG_SIZE 128
#define CHUNK_SIZE 256

#define AND(a, b) _mm_and_si128(a, b)
#define OR(a, b) _mm_or_si128(a, b)
#define XOR(a, b) _mm_xor_si128(a, b)
#define ANDN(a, b) _mm_andnot_si128(a, b)
#define NOT(a) _mm_xor_si128(ONES, a)

/* #define ADD(a,b,c) _mm_add_epi##c(a,b) */

#define MUL_SIGNED(a, b, c) _mm_mullo_epi##c(a, b)
#define ADD_SIGNED(a, b, c) _mm_add_epi##c(a, b)
#define SUB_SIGNED(a, b, c) _mm_sub_epi##c(a, b)

#define FUNC_AND __mm_and_si128_wrapper
#define FUNC_OR __mm_or_si128_wrapper
#define FUNC_XOR __mm_xor_si128_wrapper
#define FUNC_ANDN __mm_andnot_si128_wrapper
#define FUNC_ADD8 __mm_add_epi8_wrapper
#define FUNC_ADD16 __mm_add_epi16_wrapper
#define FUNC_ADD32 __mm_add_epi32_wrapper
#define FUNC_ADD64 __mm_add_epi64_wrapper
#define FUNC_SUB8 __mm_sub_epi8_wrapper
#define FUNC_SUB16 __mm_sub_epi16_wrapper
#define FUNC_SUB32 __mm_sub_epi32_wrapper
#define FUNC_SUB64 __mm_sub_epi64_wrapper
#define FUNC_MUL16 __mm_mullo_epi16_wrapper
#define FUNC_MUL32 __mm_mullo_epi32_wrapper
#define FUNC_MUL64 _mm_mullo_epi64_wrapper
#define FUNC_DIV32 __mm_div_epi32_wrapper
#define FUNC_DIV64 __mm_div_epi64_wrapper
#define SHIFT_LEFT16 __mm_sla_epi16_wrapper
#define SHIFT_LEFT32 __mm_sla_epi32_wrapper
#define SHIFT_LEFT64 __mm_sla_epi64_wrapper
#define SHIFT_RIGHT16 __mm_sra_epi16_wrapper
#define SHIFT_RIGHT32 __mm_sra_epi32_wrapper
#define SHIFT_RIGHT64 __mm_sra_epi64_wrapper
#define SHIFT_LOG_RIGHT16 __mm_srl_epi16_wrapper
#define SHIFT_LOG_RIGHT32 __mm_srl_epi32_wrapper
#define SHIFT_LOG_RIGHT64 __mm_srl_epi64_wrapper
#define SHIFT_RIGHT16F __mm_sra_epi16_wrapperF
#define SHIFT_RIGHT32F __mm_sra_epi32_wrapperF
#define SHIFT_RIGHT64F __mm_sra_epi64_wrapperF
#define SHIFT_LOG_RIGHT16F __mm_srl_epi16_wrapperF
#define SHIFT_LOG_RIGHT32F __mm_srl_epi32_wrapperF
#define SHIFT_LOG_RIGHT64F __mm_srl_epi64_wrapperF
#define SHIFT_LEFT16F __mm_sla_epi16_wrapperF
#define SHIFT_LEFT32F __mm_sla_epi32_wrapperF
#define SHIFT_LEFT64F __mm_sla_epi64_wrapperF

inline __m128i __mm_and_si128_wrapper(__m128i a, __m128i b)
{
    return _mm_and_si128(a, b);
}

inline __m128i __mm_or_si128_wrapper(__m128i a, __m128i b)
{
    return _mm_or_si128(a, b);
}

inline __m128i __mm_xor_si128_wrapper(__m128i a, __m128i b)
{
    return _mm_xor_si128(a, b);
}

inline __m128i __mm_andnot_si128_wrapper(__m128i a, __m128i b)
{
    return _mm_andnot_si128(a, b);
}

inline __m128i __mm_add_epi8_wrapper(__m128i a, __m128i b)
{
    return _mm_add_epi8(a, b);
}

inline __m128i __mm_add_epi16_wrapper(__m128i a, __m128i b)
{
    return _mm_add_epi16(a, b);
}

inline __m128i __mm_add_epi32_wrapper(__m128i a, __m128i b)
{
    return _mm_add_epi32(a, b);
}

inline __m128i __mm_add_epi64_wrapper(__m128i a, __m128i b)
{
    return _mm_add_epi64(a, b);
}

inline __m128i __mm_sub_epi8_wrapper(__m128i a, __m128i b)
{
    return _mm_sub_epi8(a, b);
}

inline __m128i __mm_sub_epi16_wrapper(__m128i a, __m128i b)
{
    return _mm_sub_epi16(a, b);
}

inline __m128i __mm_sub_epi32_wrapper(__m128i a, __m128i b)
{
    return _mm_sub_epi32(a, b);
}

inline __m128i __mm_sub_epi64_wrapper(__m128i a, __m128i b)
{
    return _mm_sub_epi64(a, b);
}

inline __m128i __mm_mullo_epi16_wrapper(__m128i a, __m128i b)
{
    return _mm_mullo_epi16(a, b);
}

inline __m128i __mm_mullo_epi32_wrapper(__m128i a, __m128i b)
{
    return _mm_mullo_epi32(a, b);
}

inline __m128i __mm_mullo_epi64_wrapper(__m128i a, __m128i b)
{
    return _mm_mullo_epi64(a, b);
}

inline __m128i __mm_div_epi32_wrapper(__m128i a, __m128i b)
{
    /* return _mm_div_epi32(a,b); */  // not implemented it seems
    return a;
}

inline __m128i __mm_div_epi64_wrapper(__m128i a, __m128i b)
{
    /* return _mm_div_epi64(a,b); */  // not implemented it seems
    return a;
}

template <int n>
inline __m128i __mm_sra_epi16_wrapper(__m128i a)
{
    return _mm_srai_epi16(a, n);
}

inline __m128i __mm_sra_epi16_wrapperF(__m128i a, int n)
{
    return _mm_srai_epi16(a, n);
}

template <int n>
inline __m128i __mm_srl_epi16_wrapper(__m128i a)
{
    return _mm_srli_epi16(a, n);
}

inline __m128i __mm_srl_epi16_wrapperF(__m128i a, int n)
{
    return _mm_srli_epi16(a, n);
}

template <int n>
inline __m128i __mm_sla_epi16_wrapper(__m128i a)
{
    return _mm_slli_epi16(a, n);
}

inline __m128i __mm_sla_epi16_wrapperF(__m128i a, int n)
{
    return _mm_slli_epi16(a, n);
}

// shifting wrapper
template <int n>
inline __m128i __mm_sra_epi32_wrapper(__m128i a)
{
    return _mm_srai_epi32(a, n);
}

inline __m128i __mm_sra_epi32_wrapperF(__m128i a, int n)
{
    return _mm_srai_epi32(a, n);
}

template <int n>
inline __m128i __mm_srl_epi32_wrapper(__m128i a)
{
    return _mm_srli_epi32(a, n);
}

inline __m128i __mm_srl_epi32_wrapperF(__m128i a, int n)
{
    return _mm_srli_epi32(a, n);
}

// shifting wrapper
template <int n>
inline __m128i __mm_sla_epi32_wrapper(__m128i a)
{
    return _mm_slli_epi32(a, n);
}

inline __m128i __mm_sla_epi32_wrapperF(__m128i a, int n)
{
    return _mm_slli_epi32(a, n);
}

template <int n>
inline __m128i __mm_sra_epi64_wrapper(__m128i a)
{
    return _mm_srai_epi64(a, n);
}

inline __m128i __mm_sra_epi64_wrapperF(__m128i a, int n)
{
    return _mm_srai_epi64(a, n);
}

template <int n>
inline __m128i __mm_srl_epi64_wrapper(__m128i a)
{
    return _mm_srli_epi64(a, n);
}

inline __m128i __mm_srl_epi64_wrapperF(__m128i a, int n)
{
    return _mm_srli_epi64(a, n);
}

// shifting wrapper
template <int n>
inline __m128i __mm_sla_epi64_wrapper(__m128i a)
{
    return _mm_slli_epi64(a, n);
}

inline __m128i __mm_sla_epi64_wrapperF(__m128i a, int n)
{
    return _mm_slli_epi64(a, n);
}

/* #define L_SHIFT(a,b,c)  _mm_slli_epi##c(a,b) */
/* #define R_SHIFT(a,b,c)  _mm_srli_epi##c(a,b) */
/* #define RA_SHIFT(a,b,c) _mm_sra_epi##c(a,_mm_set1_epi##c(b)) */

#define L_ROTATE(a, b, c)                                                                                          \
    b == 8 && c == 32    ? _mm_shuffle_epi8(a, _mm_set_epi8(14, 13, 12, 15, 10, 9, 8, 11, 6, 5, 4, 7, 2, 1, 0, 3)) \
    : b == 16 && c == 32 ? _mm_shuffle_epi8(a, _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2)) \
                         : OR(L_SHIFT(a, b, c), R_SHIFT(a, c - b, c))

#define R_ROTATE(a, b, c)                                                                                          \
    b == 8 && c == 32    ? _mm_shuffle_epi8(a, _mm_set_epi8(12, 15, 14, 13, 8, 11, 10, 9, 4, 7, 6, 5, 0, 3, 2, 1)) \
    : b == 16 && c == 32 ? _mm_shuffle_epi8(a, _mm_set_epi8(13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2)) \
                         : OR(R_SHIFT(a, b, c), L_SHIFT(a, c - b, c))

#define DATATYPE __m128i

#define SET_ALL_ONE() ONES
#define SET_ALL_ZERO() ZERO

/* Note the reverse of the pattern. */
#define PERMUT_16(a, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16) \
    _mm_shuffle_epi8(a, _mm_set_epi8(x16, x15, x14, x13, x12, x11, x10, x9, x8, x7, x6, x5, x4, x3, x2, x1))
#define PERMUT_4(a, x1, x2, x3, x4) _mm_shuffle_epi32(a, (x4 << 6) | (x3 << 4) | (x2 << 2) | x1)

#define LIFT_8(x) _mm_set1_epi8(x)
#define LIFT_16(x) _mm_set1_epi16(x)
#define LIFT_32(x) _mm_set1_epi32(x)
#define LIFT_64(x) _mm_set1_epi64x(x)

#define BITMASK(x, n, c) _mm_sub_epi##c(ZERO, __mm_and_si128(_mm_slli_epi##c(x, n), _mm_set1_epi##c(1)))

#define PACK_8x2_to_16(a, b)  /* TODO: implement with shuffles */
#define PACK_16x2_to_32(a, b) /* TODO: implement with shuffles */
#define PACK_32x2_to_64(a, b) /* TODO: implement with shuffles */

#define ORTHOGONALIZE(in, out) orthogonalize(in, out)
#define UNORTHOGONALIZE(in, out) unorthogonalize(in, out)

#define ALLOC(size) aligned_alloc(32, size * sizeof(__m128i))
#define NEW(var) new (std::align_val_t(sizeof(__m128i))) var;

/* Orthogonalization stuffs */
static uint64_t mask_l[6] = {0xaaaaaaaaaaaaaaaaUL,
                             0xccccccccccccccccUL,
                             0xf0f0f0f0f0f0f0f0UL,
                             0xff00ff00ff00ff00UL,
                             0xffff0000ffff0000UL,
                             0xffffffff00000000UL};

static uint64_t mask_r[6] = {0x5555555555555555UL,
                             0x3333333333333333UL,
                             0x0f0f0f0f0f0f0f0fUL,
                             0x00ff00ff00ff00ffUL,
                             0x0000ffff0000ffffUL,
                             0x00000000ffffffffUL};

void real_ortho(UINT_TYPE data[])
{
    for (int i = 0; i < LOG2_BITLENGTH; i++)
    {
        int nu = (1UL << i);
        for (int j = 0; j < BITLENGTH; j += (2 * nu))
            for (int k = 0; k < nu; k++)
            {
                UINT_TYPE u = data[j + k] & mask_l[i];
                UINT_TYPE v = data[j + k] & mask_r[i];
                UINT_TYPE x = data[j + nu + k] & mask_l[i];
                UINT_TYPE y = data[j + nu + k] & mask_r[i];
                data[j + k] = u | (x >> nu);
                data[j + nu + k] = (v << nu) | y;
            }
    }
}

void real_ortho_128x64(__m128i data[])
{

    __m128i mask_l[7] = {
        _mm_set1_epi64x(0xaaaaaaaaaaaaaaaaUL),
        _mm_set1_epi64x(0xccccccccccccccccUL),
        _mm_set1_epi64x(0xf0f0f0f0f0f0f0f0UL),
        _mm_set1_epi64x(0xff00ff00ff00ff00UL),
        _mm_set1_epi64x(0xffff0000ffff0000UL),
        _mm_set1_epi64x(0xffffffff00000000UL),
        _mm_set_epi64x(0x0000000000000000UL, 0xffffffffffffffffUL),

    };

    __m128i mask_r[7] = {
        _mm_set1_epi64x(0x5555555555555555UL),
        _mm_set1_epi64x(0x3333333333333333UL),
        _mm_set1_epi64x(0x0f0f0f0f0f0f0f0fUL),
        _mm_set1_epi64x(0x00ff00ff00ff00ffUL),
        _mm_set1_epi64x(0x0000ffff0000ffffUL),
        _mm_set1_epi64x(0x00000000ffffffffUL),
        _mm_set_epi64x(0xffffffffffffffffUL, 0x0000000000000000UL),
    };

    for (int i = 0; i < LOG2_BITLENGTH; i++)
    {
        int nu = (1UL << i);
        for (int j = 0; j < BITLENGTH; j += (2 * nu))
            for (int k = 0; k < nu; k++)
            {
                __m128i u = _mm_and_si128(data[j + k], mask_l[i]);
                __m128i v = _mm_and_si128(data[j + k], mask_r[i]);
                __m128i x = _mm_and_si128(data[j + nu + k], mask_l[i]);
                __m128i y = _mm_and_si128(data[j + nu + k], mask_r[i]);
                if (i <= 5)
                {
                    data[j + k] = _mm_or_si128(u, _mm_srli_epi64(x, nu));
                    data[j + nu + k] = _mm_or_si128(_mm_slli_epi64(v, nu), y);
                }
                else
                {
                    /* Note the "inversion" of srli and slli. */
                    data[j + k] = _mm_or_si128(u, _mm_slli_si128(x, 8));
                    data[j + nu + k] = _mm_or_si128(_mm_srli_si128(v, 8), y);
                }
            }
    }
}

void real_ortho_128x128(__m128i data[])
{

    __m128i mask_l[7] = {
        _mm_set1_epi64x(0xaaaaaaaaaaaaaaaaUL),
        _mm_set1_epi64x(0xccccccccccccccccUL),
        _mm_set1_epi64x(0xf0f0f0f0f0f0f0f0UL),
        _mm_set1_epi64x(0xff00ff00ff00ff00UL),
        _mm_set1_epi64x(0xffff0000ffff0000UL),
        _mm_set1_epi64x(0xffffffff00000000UL),
        _mm_set_epi64x(0x0000000000000000UL, 0xffffffffffffffffUL),

    };

    __m128i mask_r[7] = {
        _mm_set1_epi64x(0x5555555555555555UL),
        _mm_set1_epi64x(0x3333333333333333UL),
        _mm_set1_epi64x(0x0f0f0f0f0f0f0f0fUL),
        _mm_set1_epi64x(0x00ff00ff00ff00ffUL),
        _mm_set1_epi64x(0x0000ffff0000ffffUL),
        _mm_set1_epi64x(0x00000000ffffffffUL),
        _mm_set_epi64x(0xffffffffffffffffUL, 0x0000000000000000UL),
    };

    for (int i = 0; i < 7; i++)
    {
        int nu = (1UL << i);
        for (int j = 0; j < 128; j += (2 * nu))
            for (int k = 0; k < nu; k++)
            {
                __m128i u = _mm_and_si128(data[j + k], mask_l[i]);
                __m128i v = _mm_and_si128(data[j + k], mask_r[i]);
                __m128i x = _mm_and_si128(data[j + nu + k], mask_l[i]);
                __m128i y = _mm_and_si128(data[j + nu + k], mask_r[i]);
                if (i <= 5)
                {
                    data[j + k] = _mm_or_si128(u, _mm_srli_epi64(x, nu));
                    data[j + nu + k] = _mm_or_si128(_mm_slli_epi64(v, nu), y);
                }
                else
                {
                    /* Note the "inversion" of srli and slli. */
                    data[j + k] = _mm_or_si128(u, _mm_slli_si128(x, 8));
                    data[j + nu + k] = _mm_or_si128(_mm_srli_si128(v, 8), y);
                }
            }
    }
}

void real_ortho_128x128_blend(__m128i data[])
{

    __m128i mask_l[7] = {
        _mm_set1_epi64x(0xaaaaaaaaaaaaaaaaUL),
        _mm_set1_epi64x(0xccccccccccccccccUL),
        _mm_set1_epi64x(0xf0f0f0f0f0f0f0f0UL),
        _mm_set1_epi64x(0xff00ff00ff00ff00UL),
        _mm_set1_epi64x(0xffff0000ffff0000UL),
        _mm_set1_epi64x(0xffffffff00000000UL),
        _mm_set_epi64x(0UL, -1UL),

    };

    __m128i mask_r[7] = {
        _mm_set1_epi64x(0x5555555555555555UL),
        _mm_set1_epi64x(0x3333333333333333UL),
        _mm_set1_epi64x(0x0f0f0f0f0f0f0f0fUL),
        _mm_set1_epi64x(0x00ff00ff00ff00ffUL),
        _mm_set1_epi64x(0x0000ffff0000ffffUL),
        _mm_set1_epi64x(0x00000000ffffffffUL),
        _mm_set_epi64x(-1UL, 0UL),
    };

    for (int i = 0; i < 7; i++)
    {
        int nu = (1UL << i);
        for (int j = 0; j < 128; j += (2 * nu))
            for (int k = 0; k < nu; k++)
            {
                if (i <= 3)
                {
                    __m128i u = _mm_and_si128(data[j + k], mask_l[i]);
                    __m128i v = _mm_and_si128(data[j + k], mask_r[i]);
                    __m128i x = _mm_and_si128(data[j + nu + k], mask_l[i]);
                    __m128i y = _mm_and_si128(data[j + nu + k], mask_r[i]);
                    data[j + k] = _mm_or_si128(u, _mm_srli_epi64(x, nu));
                    data[j + nu + k] = _mm_or_si128(_mm_slli_epi64(v, nu), y);
                }
                else if (i == 4)
                {
                    __m128i u = data[j + k];
                    __m128i v = data[j + k];
                    __m128i x = data[j + nu + k];
                    __m128i y = data[j + nu + k];
                    data[j + k] = _mm_blend_epi16(u, _mm_srli_epi64(x, nu), 0b01010101);
                    data[j + nu + k] = _mm_blend_epi16(_mm_slli_epi64(v, nu), y, 0b01010101);
                }
                else if (i == 5)
                {
                    __m128i u = data[j + k];
                    __m128i v = data[j + k];
                    __m128i x = data[j + nu + k];
                    __m128i y = data[j + nu + k];
                    data[j + k] = _mm_blend_epi16(u, _mm_srli_epi64(x, nu), 0b00110011);
                    data[j + nu + k] = _mm_blend_epi16(_mm_slli_epi64(v, nu), y, 0b00110011);
                }
                else
                {
                    __m128i u = data[j + k];
                    __m128i v = data[j + k];
                    __m128i x = data[j + nu + k];
                    __m128i y = data[j + nu + k];
                    /* Note the "inversion" of srli and slli. */
                    data[j + k] = _mm_blend_epi16(u, _mm_slli_si128(x, 8), 0b11110000);
                    data[j + nu + k] = _mm_blend_epi16(_mm_srli_si128(v, 8), y, 0b11110000);
                }
            }
    }
}

void orthogonalize_boolean(UINT_TYPE* data, __m128i* out)
{
    /* orthogonalize_128x64(data, out); */
    for (int i = 0; i < DATTYPE; i += BITLENGTH)
        real_ortho(&(data[i]));
    for (int i = 0; i < BITLENGTH; i++)
#if BITLENGTH == 64
        out[i] = _mm_set_epi64x(data[i], data[64 + i]);

#elif BITLENGTH == 32
        out[i] = _mm_set_epi32(data[i], data[32 + i], data[64 + i], data[96 + i]);
#elif BITLENGTH == 16
        out[i] = _mm_set_epi16(
            data[i], data[16 + i], data[32 + i], data[48 + i], data[64 + i], data[80 + i], data[96 + i], data[112 + i]);
#elif BITLENGTH == 8
        out[i] = _mm_set_epi8(data[i],
                              data[8 + i],
                              data[16 + i],
                              data[24 + i],
                              data[32 + i],
                              data[40 + i],
                              data[48 + i],
                              data[56 + i],
                              data[64 + i],
                              data[72 + i],
                              data[80 + i],
                              data[88 + i],
                              data[96 + i],
                              data[104 + i],
                              data[112 + i],
                              data[120 + i]);
#endif
}

void unorthogonalize_boolean(__m128i* in, UINT_TYPE* data)
{
    for (int i = 0; i < BITLENGTH; i++)
    {
        alignas(16) UINT_TYPE tmp[DATTYPE / BITLENGTH];
        _mm_store_si128((__m128i*)tmp, in[i]);
        for (int j = 0; j < DATTYPE / BITLENGTH; j++)
            data[i + j * BITLENGTH] = tmp[DATTYPE / BITLENGTH - j - 1];
        /* data[i+j*BITLENGTH] = tmp[j]; */
    }
    for (int i = 0; i < DATTYPE; i += BITLENGTH)
        real_ortho(&(data[i]));
}
void orthogonalize_boolean_full(UINT_TYPE* data, __m128i* out)
{
    for (int i = 0; i < 128; i++)
#if BITLENGTH == 64
        out[i] = _mm_set_epi64x(data[i], data[128 + i]);
#elif BITLENGTH == 32
        out[i] = _mm_set_epi32(data[i], data[128 + i], data[256 + i], data[384 + i]);
#elif BITLENGTH == 16
        out[i] = _mm_set_epi16(data[i],
                               data[128 + i],
                               data[256 + i],
                               data[384 + i],
                               data[512 + i],
                               data[640 + i],
                               data[768 + i],
                               data[896 + i]);
#elif BITLENGTH == 8
        out[i] = _mm_set_epi8(data[i],
                              data[128 + i],
                              data[256 + i],
                              data[384 + i],
                              data[512 + i],
                              data[640 + i],
                              data[768 + i],
                              data[896 + i],
                              data[1024 + i],
                              data[1152 + i],
                              data[1280 + i],
                              data[1408 + i],
                              data[1536 + i],
                              data[1664 + i],
                              data[1792 + i],
                              data[1920 + i]);

#endif
    real_ortho_128x128(out);
}

void unorthogonalize_boolean_full(__m128i* in, UINT_TYPE* data)
{
    real_ortho_128x128(in);
    for (int i = 0; i < 128; i++)
    {
        alignas(16) UINT_TYPE tmp[DATTYPE / BITLENGTH];
        _mm_store_si128((__m128i*)tmp, in[i]);
        for (int j = 0; j < DATTYPE / BITLENGTH; j++)
            data[j * 128 + i] = tmp[DATTYPE / BITLENGTH - j - 1];
        /* data[i+j*BITLENGTH] = tmp[j]; */
    }
}

void orthogonalize_arithmetic(UINT_TYPE* in, __m128i* out, int k)
{
    for (int i = 0; i < k; i++)
#if BITLENGTH == 64
        out[i] = _mm_set_epi64x(in[i * 2 + 1], in[i * 2]);
#elif BITLENGTH == 32
        out[i] = _mm_set_epi32(in[i * 4 + 3], in[i * 4 + 2], in[i * 4 + 1], in[i * 4]);
#elif BITLENGTH == 16
        out[i] = _mm_set_epi16(in[i * 8 + 7],
                               in[i * 8 + 6],
                               in[i * 8 + 5],
                               in[i * 8 + 4],
                               in[i * 8 + 3],
                               in[i * 8 + 2],
                               in[i * 8 + 1],
                               in[i * 8]);
#elif BITLENGTH == 8
        out[i] = _mm_set_epi8(in[i * 16 + 15],
                              in[i * 16 + 14],
                              in[i * 16 + 13],
                              in[i * 16 + 12],
                              in[i * 16 + 11],
                              in[i * 16 + 10],
                              in[i * 16 + 9],
                              in[i * 16 + 8],
                              in[i * 16 + 7],
                              in[i * 16 + 6],
                              in[i * 16 + 5],
                              in[i * 16 + 4],
                              in[i * 16 + 3],
                              in[i * 16 + 2],
                              in[i * 16 + 1],
                              in[i * 16]);
#endif
}

void unorthogonalize_arithmetic(const __m128i* in, UINT_TYPE* out, int k)
{
    for (int i = 0; i < k; i++)
        _mm_store_si128((__m128i*)&(out[i * DATTYPE / BITLENGTH]), in[i]);
}

#if BITLENGTH == 8

// ReLU for 8-bit integers
__m128i relu_epi(__m128i v)
{
    __m128i zero = _mm_setzero_si128();
    return _mm_max_epi8(v, zero);
}

#elif BITLENGTH == 16

// ReLU for 16-bit integers
__m128i relu_epi(__m128i v)
{
    __m128i zero = _mm_setzero_si128();
    return _mm_max_epi16(v, zero);
}

#elif BITLENGTH == 32

// ReLU for 32-bit integers
__m128i relu_epi(__m128i v)
{
    __m128i zero = _mm_setzero_si128();
    return _mm_max_epi32(v, zero);
}

#elif BITLENGTH == 64

// ReLU for 64-bit integers (special handling)
__m128i relu_epi(__m128i v)
{
    __m128i zero = _mm_setzero_si128();
    __m128i mask = _mm_cmpgt_epi32(v, zero);  // This will compare only the lower 32 bits of each 64-bit element
    return _mm_and_si128(v, mask);
}

#endif
