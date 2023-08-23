/* ******************************************** *\
 *
 *
 *
\* ******************************************** */


/* Including headers */
#pragma once
#include <stdlib.h>
#include <x86intrin.h>
#include <stdint.h>
#include <wmmintrin.h>
#include "../config.h"

#ifndef AVX512
#define AVX512
#endif

#ifndef BITS_PER_REG
#define BITS_PER_REG 512
#endif

/* #ifdef __VAES__ */
/* #define MM_XOR _mm512_xor_si512 */
/* #define MM_AES_ENC _mm512_aesenc_epi128 */
/* #define MM_AES_DEC _mm512_aesdec_epi128 */
/* #define MM_AES_ENC_LAST _mm512_aesenclast_epi128 */
/* #define MM_AES_DEC_LAST _mm512_aesdeclast_epi128 */
/* #endif */

/* Defining 0 and 1 */
#define ZERO _mm512_setzero_si512()
#define ONES _mm512_set1_epi32(-1)

/* Defining macros */
#define REG_SIZE 512
#define CHUNK_SIZE 4096

#define AND(a,b)  _mm512_and_si512(a,b)
#define OR(a,b)   _mm512_or_si512(a,b)
#define XOR(a,b)  _mm512_xor_si512(a,b)
#define ANDN(a,b) _mm512_andnot_si512(a,b)
#define NOT(a)    _mm512_xor_si512(ONES,a)

/* #define ADD(a,b,c) _mm512_add_epi##c(a,b) */

#define ADD_SIGNED(a,b,c) _mm512_add_epi##c(a,b)
#define SUB_SIGNED(a,b,c) _mm512_sub_epi##c(a,b)
#define MUL_SIGNED(a,b,c) _mm512_mullo_epi##c(a,b)
#define MUL_SINGED_64(a,b) _mm512_mullox_epi64(a,b)

#define FUNC_AND _mm512_and_si512_wrapper
#define FUNC_OR  _mm512_or_si512_wrapper
#define FUNC_XOR _mm512_xor_si512_wrapper
#define FUNC_ADD32 _mm512_add_epi32_wrapper
#define FUNC_ADD64 _mm512_add_epi64_wrapper
#define FUNC_SUB32 _mm512_sub_epi32_wrapper
#define FUNC_SUB64 _mm512_sub_epi64_wrapper
#define FUNC_MUL32 _mm512_mullo_epi32_wrapper
#define FUNC_MUL64 _mm512_mullo_epi64_wrapper

// wrapper functions needed for some compilers

inline __m512i _mm512_and_si512_wrapper(__m512i a, __m512i b) {
  return _mm512_and_si512(a,b);
}

inline __m512i _mm512_or_si512_wrapper(__m512i a, __m512i b) {
  return _mm512_or_si512(a,b);
}

inline __m512i _mm512_xor_si512_wrapper(__m512i a, __m512i b) {
  return _mm512_xor_si512(a,b);
}

inline __m512i _mm512_add_epi32_wrapper(__m512i a, __m512i b) {
  return _mm512_add_epi32(a,b);
}

inline __m512i _mm512_add_epi64_wrapper(__m512i a, __m512i b) {
  return _mm512_add_epi64(a,b);
}

inline __m512i _mm512_sub_epi32_wrapper(__m512i a, __m512i b) {
  return _mm512_sub_epi32(a,b);
}

inline __m512i _mm512_sub_epi64_wrapper(__m512i a, __m512i b) {
  return _mm512_sub_epi64(a,b);
}

inline __m512i _mm512_mullo_epi32_wrapper(__m512i a, __m512i b) {
  return _mm512_mullo_epi32(a,b);
}

inline __m512i _mm512_mullo_epi64_wrapper(__m512i a, __m512i b) {
  return _mm512_mullo_epi64(a,b);
}



#define L_SHIFT(a,b,c)  _mm512_slli_epi##c(a,b)
#define R_SHIFT(a,b,c)  _mm512_srli_epi##c(a,b)
#define RA_SHIFT(a,b,c) _mm512_sra_epi##c(a,_mm_set1_epi##c(b))

#define L_ROTATE(a,b,c)                                                 \
  b == 8 && c == 32 ?                                                   \
    _mm512_shuffle_epi8(a,_mm512_set_epi8(14,13,12,15,10,9,8,11,6,5,4,7,2,1,0,3, \
                                          14,13,12,15,10,9,8,11,6,5,4,7,2,1,0,3, \
                                          14,13,12,15,10,9,8,11,6,5,4,7,2,1,0,3, \
                                          14,13,12,15,10,9,8,11,6,5,4,7,2,1,0,3)) : \
    b == 16 && c == 32 ?                                                \
    _mm512_shuffle_epi8(a,_mm512_set_epi8(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2, \
                                          13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2, \
                                          13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2, \
                                          13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2)) : \
    OR(L_SHIFT(a,b,c),R_SHIFT(a,c-b,c))

#define R_ROTATE(a,b,c)                                                 \
  b == 8 && c == 32 ?                                                   \
    _mm512_shuffle_epi8(a,_mm512_set_epi8(12,15,14,13,8,11,10,9,4,7,6,5,0,3,2,1, \
                                          12,15,14,13,8,11,10,9,4,7,6,5,0,3,2,1, \
                                          12,15,14,13,8,11,10,9,4,7,6,5,0,3,2,1, \
                                          12,15,14,13,8,11,10,9,4,7,6,5,0,3,2,1)) : \
    b == 16 && c == 32 ?                                                \
    _mm512_shuffle_epi8(a,_mm512_set_epi8(13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2, \
                                          13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2, \
                                          13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2, \
                                          13,12,15,14,9,8,11,10,5,4,7,6,1,0,3,2)) : \
    OR(R_SHIFT(a,b,c),L_SHIFT(a,c-b,c))


#define LIFT_8(x)  _mm512_set1_epi8(x)
#define LIFT_16(x) _mm512_set1_epi16(x)
#define LIFT_32(x) _mm512_set1_epi32(x)
#define LIFT_64(x) _mm512_set1_epi64(x)


#define BITMASK(x,n,c) _mm512_sub_epi##c(ZERO, __mm512_and_si512(_mm512_slli_epi##c(x,n), _mm512_set1_epi##c(1)))

#define PACK_8x2_to_16(a,b)  /* TODO: implement with shuffles */
#define PACK_16x2_to_32(a,b) /* TODO: implement with shuffles */
#define PACK_32x2_to_64(a,b) /* TODO: implement with shuffles */

#define DATATYPE __m512i

#define SET_ALL_ONE()  ONES
#define SET_ALL_ZERO() ZERO

/* Note: this is somewhat wrong I think */
#define PERMUT_16(a,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16) \
  _mm512_shuffle_epi8(a,_mm512_set_epi8(x16,x15,x14,x13,x12,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,\
                                        x16,x15,x14,x13,x12,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,\
                                        x16,x15,x14,x13,x12,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1,\
                                        x16,x15,x14,x13,x12,x11,x10,x9,x8,x7,x6,x5,x4,x3,x2,x1))
#define PERMUT_4(a,x1,x2,x3,x4) _mm512_shuffle_epi32(a,(x4<<6)|(x3<<4)|(x2<<2)|x1)

#define ORTHOGONALIZE(in,out) orthogonalize(in,out)
#define UNORTHOGONALIZE(in,out) unorthogonalize(in,out)

#define ALLOC(size) aligned_alloc(64,size * sizeof(__m512i))
#define NEW(var) new (std::align_val_t(sizeof(__m512i))) var;


/* Orthogonalization stuffs */
static uint64_t mask_l[6] = {
	0xaaaaaaaaaaaaaaaaUL,
	0xccccccccccccccccUL,
	0xf0f0f0f0f0f0f0f0UL,
	0xff00ff00ff00ff00UL,
	0xffff0000ffff0000UL,
	0xffffffff00000000UL
};

static uint64_t mask_r[6] = {
	0x5555555555555555UL,
	0x3333333333333333UL,
	0x0f0f0f0f0f0f0f0fUL,
	0x00ff00ff00ff00ffUL,
	0x0000ffff0000ffffUL,
	0x00000000ffffffffUL
};


void real_ortho(UINT_TYPE data[]) {
  for (int i = 0; i < LOG2_BITLENGTH; i ++) {
    int nu = (1UL << i);
    for (int j = 0; j < BITLENGTH; j += (2 * nu))
      for (int k = 0; k < nu; k ++) {
        UINT_TYPE u = data[j + k] & mask_l[i];
        UINT_TYPE v = data[j + k] & mask_r[i];
        UINT_TYPE x = data[j + nu + k] & mask_l[i];
        UINT_TYPE y = data[j + nu + k] & mask_r[i];
        data[j + k] = u | (x >> nu);
        data[j + nu + k] = (v << nu) | y;
      }
  }
}


void real_ortho_512x512(__m512i data[]) {

  __m512i mask_l[9] = {
    _mm512_set1_epi64(0xaaaaaaaaaaaaaaaaUL),
    _mm512_set1_epi64(0xccccccccccccccccUL),
    _mm512_set1_epi64(0xf0f0f0f0f0f0f0f0UL),
    _mm512_set1_epi64(0xff00ff00ff00ff00UL),
    _mm512_set1_epi64(0xffff0000ffff0000UL),
    _mm512_set1_epi64(0xffffffff00000000UL),
    _mm512_set_epi64(0UL,-1UL,0UL,-1UL,0UL,-1UL,0UL,-1UL),
    _mm512_set_epi64(0UL,0UL,-1UL,-1UL,0UL,0UL,-1UL,-1UL),
    _mm512_set_epi64(0UL,0UL,0UL,0UL,-1UL,-1UL,-1UL,-1UL),
  };

  __m512i mask_r[9] = {
    _mm512_set1_epi64(0x5555555555555555UL),
    _mm512_set1_epi64(0x3333333333333333UL),
    _mm512_set1_epi64(0x0f0f0f0f0f0f0f0fUL),
    _mm512_set1_epi64(0x00ff00ff00ff00ffUL),
    _mm512_set1_epi64(0x0000ffff0000ffffUL),
    _mm512_set1_epi64(0x00000000ffffffffUL),
    _mm512_set_epi64(-1UL,0UL,-1UL,0UL,-1UL,0UL,-1UL,0UL),
    _mm512_set_epi64(-1UL,-1UL,0UL,0UL,-1UL,-1UL,0UL,0UL),
    _mm512_set_epi64(-1UL,-1UL,-1UL,-1UL,0UL,0UL,0UL,0UL),
  };

  for (int i = 0; i < 9; i ++) {
    int nu = (1UL << i);
    for (int j = 0; j < 512; j += (2 * nu))
      for (int k = 0; k < nu; k ++) {
        __m512i u = _mm512_and_si512(data[j + k], mask_l[i]);
        __m512i v = _mm512_and_si512(data[j + k], mask_r[i]);
        __m512i x = _mm512_and_si512(data[j + nu + k], mask_l[i]);
        __m512i y = _mm512_and_si512(data[j + nu + k], mask_r[i]);
        if (i <= 5) {
          data[j + k] = _mm512_or_si512(u, _mm512_srli_epi64(x, nu));
          data[j + nu + k] = _mm512_or_si512(_mm512_slli_epi64(v, nu), y);
        } else if (i == 6) {
          /* Note the "inversion" of srli and slli. */
          data[j + k] = _mm512_or_si512(u, _mm512_bslli_epi128(x, 8));
          data[j + nu + k] = _mm512_or_si512(_mm512_bsrli_epi128(v, 8), y);
        } else if (i == 7) {
          /* might be 0b01001110 instead */
          data[j + k] = _mm512_or_si512(u, _mm512_permutex_epi64(x,0b10110001));
          data[j + nu + k] = _mm512_or_si512(_mm512_permutex_epi64(v,0b10110001), y);
        } else {
          /* might be 0,1,2,3,4,5,6,7 */
          __m512i ctrl = _mm512_set_epi64(4,5,6,7,0,1,2,3);
          data[j + k] = _mm512_or_si512(u, _mm512_permutexvar_epi64(ctrl,x));
          data[j + nu + k] = _mm512_or_si512(_mm512_permutexvar_epi64(ctrl,v), y);
        }
      }
  }
}

void orthogonalize_boolean(UINT_TYPE* data, __m512i* out) {
  for (int i = 0; i < DATTYPE/BITLENGTH; i++)
      real_ortho(&(data[i*BITLENGTH]));
  for (int i = 0; i < BITLENGTH; i++)
#if BITLENGTH == 64
  out[i] = _mm512_set_epi64(data[i], data[64+i], data[128+i], data[192+i], data[256+i], data[320+i], data[384+i], data[448+i]);
#elif BITLENGTH == 32
    out[i] = _mm512_set_epi32(data[i], data[32+i], data[64+i], data[96+i], data[128+i], data[160+i], data[192+i], data[224+i], data[256+i], data[288+i], data[320+i], data[352+i], data[384+i], data[416+i], data[448+i], data[480+i]);
#elif BITLENGTH == 16
    out[i] = _mm512_set_epi16(data[i], data[16+i], data[32+i], data[48+i], data[64+i], data[80+i], data[96+i], data[112+i], data[128+i], data[144+i], data[160+i], data[176+i], data[192+i], data[208+i], data[224+i], data[240+i], data[256+i], data[272+i], data[288+i], data[304+i], data[320+i], data[336+i], data[352+i], data[368+i], data[384+i], data[400+i], data[416+i], data[432+i], data[448+i], data[464+i], data[480+i], data[496+i]);
#endif
}

void orthogonalize_boolean_full(UINT_TYPE* data, __m512i* out) {
  for (int i = 0; i < 512; i++)
#if BITLENGTH == 64
    out[i] = _mm512_set_epi64(data[i], data[512+i], data[1024+i], data[1536+i], data[2048+i], data[2560+i], data[3072+i], data[3584+i]);
#elif BITLENGTH == 32
    out[i] = _mm512_set_epi32(data[i], data[512+i], data[1024+i], data[1536+i], data[2048+i], data[2560+i], data[3072+i], data[3584+i], data[4096+i], data[4608+i], data[5120+i], data[5632+i], data[6144+i], data[6656+i], data[7168+i], data[7680+i]);
#elif BITLENGTH == 16
    out[i] = _mm512_set_epi16(data[i], data[512+i], data[1024+i], data[1536+i], data[2048+i], data[2560+i], data[3072+i], data[3584+i], data[4096+i], data[4608+i], data[5120+i], data[5632+i], data[6144+i], data[6656+i], data[7168+i], data[7680+i], data[8192+i], data[8704+i], data[9216+i], data[9728+i], data[10240+i], data[10752+i], data[11264+i], data[11776+i], data[12288+i], data[12800+i], data[13312+i], data[13824+i], data[14336+i], data[14848+i], data[15360+i], data[15872+i]);
#endif
  real_ortho_512x512(out);
}

void orthogonalize_arithmetic(UINT_TYPE *in, __m512i *out) {
  for (int i = 0; i < BITLENGTH; i++)
#if BITLENGTH == 64
    out[i] = _mm512_set_epi64(in[i*8+7], in[i*8+6], in[i*8+5], in[i*8+4],
                              in[i*8+3], in[i*8+2], in[i*8+1], in[i*8]);
#elif BITLENGTH == 32
    out[i] = _mm512_set_epi32(in[i*16+15], in[i*16+14], in[i*16+13], in[i*16+12],
                              in[i*16+11], in[i*16+10], in[i*16+9], in[i*16+8],
                              in[i*16+7], in[i*16+6], in[i*16+5], in[i*16+4],
                              in[i*16+3], in[i*16+2], in[i*16+1], in[i*16]);
#elif BITLENGTH == 16
    out[i] = _mm512_set_epi16(in[i*32+31], in[i*32+30], in[i*32+29], in[i*32+28],
                              in[i*32+27], in[i*32+26], in[i*32+25], in[i*32+24],
                              in[i*32+23], in[i*32+22], in[i*32+21], in[i*32+20],
                              in[i*32+19], in[i*32+18], in[i*32+17], in[i*32+16],
                              in[i*32+15], in[i*32+14], in[i*32+13], in[i*32+12],
                              in[i*32+11], in[i*32+10], in[i*32+9], in[i*32+8],
                              in[i*32+7], in[i*32+6], in[i*32+5], in[i*32+4],
                              in[i*32+3], in[i*32+2], in[i*32+1], in[i*32]);
#endif
}

void orthogonalize_arithmetic_full(UINT_TYPE *in, __m512i *out) {
  for (int i = 0; i < 512; i++)
#if BITLENGTH == 64
    out[i] = _mm512_set_epi64(in[i*8+7], in[i*8+6], in[i*8+5], in[i*8+4],
                              in[i*8+3], in[i*8+2], in[i*8+1], in[i*8]);
#elif BITLENGTH == 32
    out[i] = _mm512_set_epi32(in[i*16+15], in[i*16+14], in[i*16+13], in[i*16+12],
                              in[i*16+11], in[i*16+10], in[i*16+9], in[i*16+8],
                              in[i*16+7], in[i*16+6], in[i*16+5], in[i*16+4],
                              in[i*16+3], in[i*16+2], in[i*16+1], in[i*16]);
#elif BITLENGTH == 16
    out[i] = _mm512_set_epi16(in[i*32+31], in[i*32+30], in[i*32+29], in[i*32+28],
                              in[i*32+27], in[i*32+26], in[i*32+25], in[i*32+24],
                              in[i*32+23], in[i*32+22], in[i*32+21], in[i*32+20],
                              in[i*32+19], in[i*32+18], in[i*32+17], in[i*32+16],
                              in[i*32+15], in[i*32+14], in[i*32+13], in[i*32+12],
                              in[i*32+11], in[i*32+10], in[i*32+9], in[i*32+8],
                              in[i*32+7], in[i*32+6], in[i*32+5], in[i*32+4],
                              in[i*32+3], in[i*32+2], in[i*32+1], in[i*32]);
#endif
}

void unorthogonalize_boolean(__m512i *in, UINT_TYPE* data) {
  for (int i = 0; i < BITLENGTH; i++) {
    alignas(64) uint64_t tmp[DATTYPE/BITLENGTH];
    _mm512_store_si512 ((__m512i*)tmp, in[i]);
    for (int j = 0; j < DATTYPE/BITLENGTH; j++)
      /* data[j*BITLENGTH+i] = tmp[j]; */
      data[i+j*BITLENGTH] = tmp[DATTYPE/BITLENGTH-j-1];
  }
  for (int i = 0; i < DATTYPE/BITLENGTH; i++)
      real_ortho(&(data[i*BITLENGTH]));
}

void unorthogonalize_boolean_full(__m512i *in, UINT_TYPE* data) {
  real_ortho_512x512(in);
  for (int i = 0; i < 512; i++) {
    alignas(64) UINT_TYPE tmp[DATTYPE/BITLENGTH];
    _mm512_store_si512 ((__m512i*)tmp, in[i]);
    for (int j = 0; j < DATTYPE/BITLENGTH; j++)
      /* data[j*512+i] = tmp[j]; */
      data[j*512+i] = tmp[DATTYPE/BITLENGTH-j-1];
  }
}

void unorthogonalize_arithmetic(__m512i *in, UINT_TYPE *out) {
  for (int i = 0; i < BITLENGTH; i++)
    _mm512_store_si512 ((__m512i*)&(out[i*DATTYPE/BITLENGTH]), in[i]);
}

void unorthogonalize_arithmetic_full(__m512i *in, UINT_TYPE *out) {
  for (int i = 0; i < 512; i++)
    _mm512_store_si512 ((__m512i*)&(out[i*DATTYPE/BITLENGTH]), in[i]);
}
