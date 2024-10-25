#pragma once
#include <cstdint>
#include <string>
#if FAKE_TRIPLES == 0
#define generateArithmeticTriples generateLibOTArithmeticTriples
#define generateBooleanTriples generateLibOTBooleanTriples

// Input: unitialized arrays of arithmetic triple shares [a], [b], [c] with size num_triples and ring size of bitlength
// Input: ip and port of the other party to connect to
// Output: [a], [b], [c] will be filled with triples
template <typename type>
void generateArithmeticLibOTTriples(type a[], type b[], type c[], int bitlength, uint64_t num_triples, std::string ip, int port);

// Input: unitialized arrays of boolean triple shares [a], [b], [c] with size num_triples and ring size of bitlength
// Input: ip and port of the other party to connect to
// Output: [a], [b], [c] will be filled with triples
template <typename type>
void generateBooleanLibOTTriples(type a[], type b[], type c[], int bitlength, uint64_t num_triples, std::string ip, int port);

#else

#define generateArithmeticTriples generateFakeArithmeticTriples
#define generateBooleanTriples generateFakeBooleanTriples




template <typename type>
void generateFakeArithmeticTriples(type a[], type b[], type c[], int bitlength, uint64_t num_triples, std::string ip, int port)
{
    num_triples /= DATTYPE/BITLENGTH;
    for (uint64_t i = 0; i < num_triples; i++)
    {
/* #if num_players == 2 */
#if PARTY == 0
        a[i] = PROMOTE(3);
        b[i] = PROMOTE(5);
        c[i] = PROMOTE(34);
#else
        a[i] = PROMOTE(5);
        b[i] = PROMOTE(2);
        c[i] = PROMOTE(22);
#endif
/* #else */
        /* a[i] = SET_ALL_ZERO(); */
        /* b[i] = SET_ALL_ZERO(); */
        /* c[i] = SET_ALL_ZERO(); */
        /* a[i] = PROMOTE(2); */
        /* b[i] = PROMOTE(3); */
        /* c[i] = PROMOTE(24); */
/* #endif */
    }
}

template <typename type>
void generateFakeBooleanTriples(type a[], type b[], type c[], int bitlength, uint64_t num_triples, std::string ip, int port)
{
    num_triples /= DATTYPE;
    for (uint64_t i = 0; i < num_triples; i++)
    {
#if num_paties == 2
#if PARTY == 0
        a[i] = SET_ALL_ZERO();
        b[i] = SET_ALL_ONE();
        c[i] = SET_ALL_ZERO();
#else
        a[i] = SET_ALL_ONE();
        b[i] = SET_ALL_ONE();
        c[i] = SET_ALL_ONE();
#endif
#else
        a[i] = SET_ALL_ZERO();
        b[i] = SET_ALL_ZERO();
        c[i] = SET_ALL_ZERO();
#endif
    }
}

#endif

    
