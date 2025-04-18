#pragma once
#include "../include/pch.h"

#ifdef __BMI2__
#include <immintrin.h>
#include <x86intrin.h>

void pack(bool* orig, uint8_t* dest, size_t l)
{
    size_t s = 0;
    for (size_t i = 0; i < l; i += 8)
    {
        bool* address = &orig[i];
        dest[s] = _pext_u64(*((uint64_t*)address), 0x0101010101010101ULL);
        s += 1;
    }
}

void unpack(uint8_t* orig, bool* dest, size_t l)
{
    size_t s = 0;
    for (size_t i = 0; i < l; i += 8)
    {
        dest[i] = _pdep_u64(orig[s], 0x0101010101010101ULL);
        s += 1;
    }
}

#else

uint8_t pack8bools(bool* a)
{
    uint64_t t;
    memcpy(&t, a, sizeof t);  //  strict-aliasing & alignment safe load
    return 0x8040201008040201ULL * t >> 56;
    // bit order: a[0]<<7 | a[1]<<6 | ... | a[7]<<0  on little-endian
    // for a[0] => LSB, use 0x0102040810204080ULL    on little-endian
}

void unpack8bools(uint8_t b, bool* a)
{
    // on little-endian,  a[0] = (b>>7) & 1  like printing order
    auto MAGIC = 0x8040201008040201ULL;  // for opposite order, byte-reverse this
    auto MASK = 0x8080808080808080ULL;
    uint64_t t = ((MAGIC * b) & MASK) >> 7;
    memcpy(a, &t, sizeof t);  // store 8 bytes without UB
}

void pack(bool* orig, uint8_t* dest, size_t l)
{
    size_t s = 0;
    for (size_t i = 0; i < l; i += 8)
    {
        dest[s] = pack8bools(&orig[i]);
        s += 1;
    }
}
void unpack(uint8_t* orig, bool* dest, size_t l)
{
    size_t s = 0;
    for (size_t i = 0; i < l; i += 8)
    {
        unpack8bools(orig[s], &dest[i]);
        s += 1;
    }
}

/* void unpack(char* orig, bool* dest, size_t l) */
/* { */
/*     for(size_t i = 0; i < l; i++) */
/*     { */
/*         char mask = 1 << (i % 8); */
/*         dest[i] = mask & orig[l/8]; */
/*     } */
/* } */

/* void pack(bool* orig, char* dest, size_t l) */
/* { */
/*     for(size_t i = 0; i < l; i++) */
/*     { */
/*             char mask = 1 << (i % 8); */
/*             dest[i/8] += orig[i] * mask; */
/*     } */
/* } */
#endif
