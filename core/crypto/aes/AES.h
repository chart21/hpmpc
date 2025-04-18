#ifndef __AES_NI_H__
#define __AES_NI_H__
#include <stdint.h>  //for int8_t
#include <string.h>  //for memcmp
// compile using gcc and following arguments: -g;-O0;-Wall;-msse2;-msse;-march=native;-maes
#include "../../arch/DATATYPE.h"
// internal stuff

#if defined(__AVX512F__) && defined(__VAES__)
#include <wmmintrin.h>  //for intrinsics for AES-NI
#include <x86intrin.h>
#define AES_DATTYPE 512
#define AES_TYPE __m512i
#define MM_XOR(a, b) _mm512_xor_si512(a, b)
#define MM_AES_ENC(a, b) _mm512_aesenc_epi128(a, b)
#define MM_AES_ENC_LAST(a, b) _mm512_aesenclast_epi128(a, b)
#define MM_AES_STORE(a, b) _mm512_store_si512(a, b)
#elif defined(__AVX2__) && defined(__VAES__)
#include <wmmintrin.h>  //for intrinsics for AES-NI
#include <x86intrin.h>
#define AES_DATTYPE 256
#define AES_TYPE __m256i
#define MM_XOR(a, b) _mm256_xor_si256(a, b)
#define MM_AES_ENC(a, b) _mm256_aesenc_epi128(a, b)
#define MM_AES_ENC_LAST(a, b) _mm256_aesenclast_epi128(a, b)
#define MM_AES_STORE(a, b) _mm256_store_si256(a, b)
#elif defined(__AES__)
#include <wmmintrin.h>  //for intrinsics for AES-NI
#include <x86intrin.h>
#define AES_DATTYPE 128
#define AES_TYPE __m128i
#define MM_XOR(a, b) _mm_xor_si128(a, b)
#define MM_AES_ENC(a, b) _mm_aesenc_si128(a, b)
#define MM_AES_ENC_LAST(a, b) _mm_aesenclast_si128(a, b)
#define MM_AES_STORE(a, b) _mm_store_si128(a, b)
#else
#ifndef CONFIG_ERROR
#define CONFIG_ERROR
#endif
#endif
// macros
#define DO_ENC_BLOCK(m, k)             \
    do                                 \
    {                                  \
        m = MM_XOR(m, k[0]);           \
        m = MM_AES_ENC(m, k[1]);       \
        m = MM_AES_ENC(m, k[2]);       \
        m = MM_AES_ENC(m, k[3]);       \
        m = MM_AES_ENC(m, k[4]);       \
        m = MM_AES_ENC(m, k[5]);       \
        m = MM_AES_ENC(m, k[6]);       \
        m = MM_AES_ENC(m, k[7]);       \
        m = MM_AES_ENC(m, k[8]);       \
        m = MM_AES_ENC(m, k[9]);       \
        m = MM_AES_ENC_LAST(m, k[10]); \
    } while (0)

template <typename Datatype>
void AES_enc(Datatype& m, const Datatype* k)
{
    m = MM_XOR(m, k[0]);
    m = MM_AES_ENC(m, k[1]);
    m = MM_AES_ENC(m, k[2]);
    m = MM_AES_ENC(m, k[3]);
    m = MM_AES_ENC(m, k[4]);
    m = MM_AES_ENC(m, k[5]);
    m = MM_AES_ENC(m, k[6]);
    m = MM_AES_ENC(m, k[7]);
    m = MM_AES_ENC(m, k[8]);
    m = MM_AES_ENC(m, k[9]);
    m = MM_AES_ENC_LAST(m, k[10]);
}

#define DO_DEC_BLOCK(m, k)            \
    do                                \
    {                                 \
        m = MM_XOR(m, k[10 + 0]);     \
        m = MM_AES_DEC(m, k[10 + 1]); \
        m = MM_AES_DEC(m, k[10 + 2]); \
        m = MM_AES_DEC(m, k[10 + 3]); \
        m = MM_AES_DEC(m, k[10 + 4]); \
        m = MM_AES_DEC(m, k[10 + 5]); \
        m = MM_AES_DEC(m, k[10 + 6]); \
        m = MM_AES_DEC(m, k[10 + 7]); \
        m = MM_AES_DEC(m, k[10 + 8]); \
        m = MM_AES_DEC(m, k[10 + 9]); \
        m = MM_AES_DEC_LAST(m, k[0]); \
    } while (0)

#if defined(__AES__)
#define AES_128_key_exp(k, rcon) aes_128_key_expansion(k, _mm_aeskeygenassist_si128(k, rcon))

static __m128i aes_128_key_expansion(__m128i key, __m128i keygened)
{
    keygened = _mm_shuffle_epi32(keygened, _MM_SHUFFLE(3, 3, 3, 3));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    key = _mm_xor_si128(key, _mm_slli_si128(key, 4));
    return _mm_xor_si128(key, keygened);
}

static void aes128_load_key_enc_only(uint8_t* enc_key, __m128i* key_schedule)
{
    key_schedule[0] = _mm_loadu_si128((const __m128i*)enc_key);
    key_schedule[1] = AES_128_key_exp(key_schedule[0], 0x01);
    key_schedule[2] = AES_128_key_exp(key_schedule[1], 0x02);
    key_schedule[3] = AES_128_key_exp(key_schedule[2], 0x04);
    key_schedule[4] = AES_128_key_exp(key_schedule[3], 0x08);
    key_schedule[5] = AES_128_key_exp(key_schedule[4], 0x10);
    key_schedule[6] = AES_128_key_exp(key_schedule[5], 0x20);
    key_schedule[7] = AES_128_key_exp(key_schedule[6], 0x40);
    key_schedule[8] = AES_128_key_exp(key_schedule[7], 0x80);
    key_schedule[9] = AES_128_key_exp(key_schedule[8], 0x1B);
    key_schedule[10] = AES_128_key_exp(key_schedule[9], 0x36);
}

#endif

#if defined(__AVX512F__) && defined(__VAES__)
// generate 4 round keys with aes128_load_key_enc_only and pack them into 512-bit vector
static void aes128_load_key_enc_only_512(uint8_t* enc_key, __m512i* key_schedule)
{
    alignas(AES_DATTYPE / 8) __m128i key_schedule_128[4][11];
    for (int i = 0; i < 4; i++)
    {
        aes128_load_key_enc_only(enc_key + i * 16, key_schedule_128[i]);
    }
    alignas(AES_DATTYPE / 8) uint64_t key_schedule_64[4][11][2];
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            key_schedule_64[i][j][0] = _mm_extract_epi64(key_schedule_128[i][j], 0);
            key_schedule_64[i][j][1] = _mm_extract_epi64(key_schedule_128[i][j], 1);
        }
    }
    for (int i = 0; i < 11; i++)
    {
        key_schedule[i] = _mm512_set_epi64(key_schedule_64[3][i][1],
                                           key_schedule_64[3][i][0],
                                           key_schedule_64[2][i][1],
                                           key_schedule_64[2][i][0],
                                           key_schedule_64[1][i][1],
                                           key_schedule_64[1][i][0],
                                           key_schedule_64[0][i][1],
                                           key_schedule_64[0][i][0]);
    }
}

#elif defined(__AVX2__) && defined(__VAES__)
// generate 2 round keys with aes128_load_key_enc_only and pack them into 256-bit vector
static void aes128_load_key_enc_only_256(uint8_t* enc_key, __m256i* key_schedule)
{
    alignas(AES_DATTYPE / 8) __m128i key_schedule_128[2][11];
    for (int i = 0; i < 2; i++)
    {
        aes128_load_key_enc_only(enc_key + i * 16, key_schedule_128[i]);
    }
    alignas(AES_DATTYPE / 8) uint64_t key_schedule_64[2][11][2];
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 11; j++)
        {
            key_schedule_64[i][j][0] = _mm_extract_epi64(key_schedule_128[i][j], 0);
            key_schedule_64[i][j][1] = _mm_extract_epi64(key_schedule_128[i][j], 1);
        }
    }
    for (int i = 0; i < 11; i++)
    {
        key_schedule[i] = _mm256_set_epi64x(
            key_schedule_64[1][i][1], key_schedule_64[1][i][0], key_schedule_64[0][i][1], key_schedule_64[0][i][0]);
    }
}
#endif

#if defined(__AVX512F__) && defined(__VAES__)
// public api
static void aes_load_enc(uint8_t* enc_key, __m512i* key_schedule)
{
    aes128_load_key_enc_only_512(enc_key, key_schedule);
}

#elif defined(__AVX2__) && defined(__VAES__)
// public api
static void aes_load_enc(uint8_t* enc_key, __m256i* key_schedule)
{
    aes128_load_key_enc_only_256(enc_key, key_schedule);
}
#elif defined(__AES__)
// public api
static void aes_load_enc(uint8_t* enc_key, __m128i* key_schedule)
{
    aes128_load_key_enc_only(enc_key, key_schedule);
}
#endif
#endif
