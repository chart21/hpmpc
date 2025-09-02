#pragma once
#include "include/pch.h"
#include "arch/DATATYPE.h"
#if FAKE_TRIPLES == 0
#define generateArithmeticTriples generateArithmeticDummyTriples
#define generateBooleanTriples generateBooleanDummyTriples
#define generateArithmeticAB2Triples generateArithmeticAB2DummyTriples
#define generateBooleanAB2Triples generateBooleanAB2DummyTriples

// Input: arrays of arithmetic triple shares [a], [b], [c] with size num_triples and ring size of bitlength
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateArithmeticDummyTriples(type a[],
                                    type b[],
                                    type c[],
                                    int bitlength,
                                    uint64_t num_triples,
                                    std::string ip,
                                    int port)
{
    if(num_triples == 0) return;

    //convert SIMD variables to regular uints
    const int vectorization_factor = DATTYPE / bitlength; 
    UINT_TYPE* uint_a = NEW(UINT_TYPE[num_triples]);
    unorthogonalize_arithmetic(a, uint_a, num_triples / (vectorization_factor)); 
    UINT_TYPE* uint_b = NEW(UINT_TYPE[num_triples]);
    unorthogonalize_arithmetic(b, uint_b, num_triples / (vectorization_factor));
    UINT_TYPE* uint_c = NEW(UINT_TYPE[num_triples]);
    
    for (uint64_t i = 0; i < num_triples; i++)
    {
       uint_c[i] = uint_a[i] + uint_b[i]; // dummy assignment, replace with actual triple generation
    }
    
    // convert UINT triple to SIMD type
    orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
    DELETEARR(uint_a);
    DELETEARR(uint_b);
    DELETEARR(uint_c);
}

// Input: array of boolean triple shares [a], [b], [c] with size num_triples
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateBooleanDummyTriples(type a[],
                                 type b[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port)
{
    if(num_triples == 0) return;
    //reinterpret SIMD bitstream as uint8 bitstream
    uint8_t* uint_a = (uint8_t*) a;
    uint8_t* uint_b = (uint8_t*) b;
    uint8_t* uint_c = (uint8_t*) c;
    
    for (uint64_t i = 0; i < num_triples / 8; i++)
    {
       uint_c[i] = uint_a[i] ^ uint_b[i]; // dummy assignment, replace with actual triple generation
    }

}


// Input: arrays of arithmetic triple shares [a], [b], [c] with size num_triples and ring size of bitlength
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateArithmeticAB2DummyTriples(type a[],
                                    type b[],
                                    type c[],
                                    int bitlength,
                                    uint64_t num_triples,
                                    std::string ip,
                                    int port)
{
    if(num_triples == 0) return;

    //convert SIMD variables to regular uints
    const int vectorization_factor = DATTYPE / bitlength; 
    UINT_TYPE* uint_a = NEW(UINT_TYPE[num_triples]);
    unorthogonalize_arithmetic(a, uint_a, num_triples / (vectorization_factor)); 
#if PARTY == 1
    UINT_TYPE* uint_b = NEW(UINT_TYPE[num_triples]);
    unorthogonalize_arithmetic(b, uint_b, num_triples / (vectorization_factor));
#endif // P_0 doesn't need b for AB2
    UINT_TYPE* uint_c = NEW(UINT_TYPE[num_triples]);
    
    for (uint64_t i = 0; i < num_triples; i++)
    {
#if PARTY == 1
       uint_c[i] = uint_a[i] + uint_b[i]; // dummy assignment, replace with actual triple generation
#else 
       uint_c[i] = uint_a[i]; // dummy assignment, replace with actual triple generation 
#endif 
    }
    
    // convert UINT triple to SIMD type
    orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
    DELETEARR(uint_a);
#if PARTY == 1
    DELETEARR(uint_b);
#endif
    DELETEARR(uint_c);
}

// Input: array of boolean triple shares [a], [b], [c] with size num_triples
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateBooleanAB2DummyTriples(type a[],
                                 type b[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port)
{
    if(num_triples == 0) return;
    //reinterpret SIMD bitstream as uint8 bitstream
    uint8_t* uint_a = (uint8_t*) a;
#if PARTY == 1
    uint8_t* uint_b = (uint8_t*) b;
#endif // P_0 doesn't need b for AB2
    uint8_t* uint_c = (uint8_t*) c;
    
    for (uint64_t i = 0; i < num_triples / 8; i++)
    {
#if PARTY == 1
       uint_c[i] = uint_a[i] ^ uint_b[i]; // dummy assignment, replace with actual triple generation
#else
       uint_c[i] = uint_a[i]; // dummy assignment, replace with actual triple generation 
#endif 
    }

}

#else

#define generateArithmeticTriples generateFakeArithmeticTriples
#define generateBooleanTriples generateFakeBooleanTriples
#define generateArithmeticAB2Triples generateFakeArithmeticTriples
#define generateBooleanAB2Triples generateFakeBooleanTriples

template <typename type>
void generateFakeArithmeticTriples(type a[],
                                   type b[],
                                   type c[],
                                   int bitlength,
                                   uint64_t num_triples,
                                   std::string ip,
                                   int port)
{
}

template <typename type>
void generateFakeBooleanTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

    template <typename type>
void generateFakeAB2ArithmeticTriples(type a[],
                                   type b[],
                                   type c[],
                                   int bitlength,
                                   uint64_t num_triples,
                                   std::string ip,
                                   int port)
{
}

template <typename type>
void generateFakeAB2BooleanTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

#endif
