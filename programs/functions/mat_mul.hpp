#pragma once
#include "../../protocols/Protocols.h"
#include "../../protocols/XOR_Share.hpp"
#include "../../protocols/Additive_Share.hpp"
#include "../../protocols/Matrix_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
/* #include "boolean_adder.hpp" */
#include "boolean_adder_bandwidth.hpp"

#include "boolean_adder_msb.hpp"
#include "ppa_msb.hpp"
#include "ppa.hpp"
#include "ppa_msb_unsafe.hpp"
#include "ppa_msb_4_way.hpp"

#include "../../utils/print.hpp"
/* #if USE_CUDA_GEMM == 1 */
/* #include "../../cuda/gemm_cutlass_int.h" */
/* #endif */
#include <cmath>

/* #include "boolean_adder.hpp" */
/* #include "ppa.hpp" */
#if FUNCTION_IDENTIFIER == 19
#define FUNCTION AND_bench
#elif FUNCTION_IDENTIFIER == 18
#define FUNCTION fixed_test
#elif FUNCTION_IDENTIFIER == 13
#define FUNCTION dot_prod_bench
#elif FUNCTION_IDENTIFIER == 14
#define FUNCTION dot_prod_bench
#elif FUNCTION_IDENTIFIER == 15
#define FUNCTION conv2D
#elif FUNCTION_IDENTIFIER == 21
#define FUNCTION testo_functions
#elif FUNCTION_IDENTIFIER == 22 || FUNCTION_IDENTIFIER == 23
#define FUNCTION MULT_Round_Test
#elif FUNCTION_IDENTIFIER == 24
#define FUNCTION A2Bit_Setup_Round_Test
#elif FUNCTION_IDENTIFIER == 25
#define FUNCTION BIT2A_Setup_Round_Test
#elif FUNCTION_IDENTIFIER == 26
#define FUNCTION dot_prod_round_bench
/* #define FUNCTION FC_bench */
/* #define USE_EIGEN 1 */
#elif FUNCTION_IDENTIFIER == 27
#define FUNCTION dot_prod_eigen_bench
#define USE_EIGEN 1
#elif FUNCTION_IDENTIFIER == 28
#define FUNCTION argmax_test
#elif FUNCTION_IDENTIFIER == 29 || FUNCTION_IDENTIFIER == 30
#define FUNCTION mult34_test
#elif FUNCTION_IDENTIFIER == 31 || FUNCTION_IDENTIFIER == 32 || FUNCTION_IDENTIFIER == 33 || FUNCTION_IDENTIFIER == 34
#define FUNCTION dot234_test
#elif FUNCTION_IDENTIFIER == 35
#define FUNCTION RELU_range_test
/* #elif FUNCTION_IDENTIFIER == 39 */
/* #include "comp_trunc.hpp" */
/* #define FUNCTION test_comp_trunc */
#elif FUNCTION_IDENTIFIER == 39
#define FUNCTION mat_mul_test
#endif
#define RESULTTYPE DATATYPE
#if FRACTIONAL > 0

template <typename float_type, typename uint_type, size_t fractional>
float_type fixedToFloat(uint_type val) {
    static_assert(std::is_integral<uint_type>::value, "uint_type must be an integer type");
    static_assert(fractional <= (sizeof(uint_type) * 8 - 1), "fractional bits are too large for the uint_type");

    using sint_type = typename std::make_signed<uint_type>::type;
    float_type scaleFactor = static_cast<float_type>(1ULL << fractional);
    return static_cast<float_type>(static_cast<sint_type>(val)) / scaleFactor;
}

template <typename float_type, typename uint_type, size_t fractional>
uint_type floatToFixed(float_type val) {
    static_assert(std::is_integral<uint_type>::value, "uint_type must be an integer type");
    static_assert(fractional <= (sizeof(uint_type) * 8 - 1), "fractional bits are too large for the uint_type");

    bool isNegative = (val < 0);
    /* if (isNegative) val = -val; // Make it positive for easier handling */
    /* // Split into integer and fractional parts */
    /* uint_type intPart = static_cast<uint_type>(val); */
    /* float_type fracPart = val - intPart; */
    
    // Split into integer and fractional parts
    uint_type intPart = static_cast<uint_type>(std::abs(val));  // Taking absolute value here
    float_type fracPart = std::abs(val) - intPart;  // Taking absolute value here too


    // Convert fractional part
    fracPart *= static_cast<float_type>(1ULL << fractional);
    uint_type fracInt = static_cast<uint_type>(fracPart + 0.5); // Adding 0.5 for rounding

    // Combine
    uint_type result = (intPart << fractional) | fracInt;

    // Apply two's complement if negative
    if (isNegative) {
        result = ~result + 1;
    }

    return result;
}

#endif

#if FUNCTION_IDENTIFIER == 21
#include "Relu.hpp"
template<typename Share>
void testo_functions(DATATYPE* res)
{
using A = Additive_Share<DATATYPE, Share>;
const int num_inputs = 100;
auto a = new A[num_inputs];
auto b = new A[num_inputs];
auto c = new A[num_inputs];
Share::communicate(); // dummy round
for(int i = 0; i < num_inputs; i++)
{
    c[0] = a[i].prepare_dot3(a[i],b[i]);
    /* c[0] = a[i].prepare_dot4(a[i],b[i],c[0]); */
}
c[0].mask_and_send_dot();
Share::communicate();
c[0].complete_mult();
Share::communicate();
c[0].mask_and_send_dot_without_trunc();
Share::communicate();
c[0].complete_mult_without_trunc();
Share::communicate();
/* c[0].prepare_mult_public_fixed(1); */
/* Share::communicate(); */
/* c[0].complete_public_mult_fixed(); */
/* Share::communicate(); */
/* c[0] = a[0].prepare_mult(b[0]); */
/* Share::communicate(); */
/* c[0].complete_mult_without_trunc(); */
/* RELU(a,a+num_inputs,c); */
Share::communicate();

}

#endif

#if FUNCTION_IDENTIFIER == 22 || FUNCTION_IDENTIFIER == 23 

template<typename Share>
void MULT_Round_Test(DATATYPE* res)
{
using A = Additive_Share<DATATYPE, Share>;
const int sequential_rounds = NUM_INPUTS; //specify number of rounds 
auto a = new A[DATTYPE];
auto b = new A[DATTYPE];

Share::communicate(); // dummy round
for(int i = 0; i < sequential_rounds; i++)
{
    for(int i = 0; i < DATTYPE; i++)
    {
#if FUNCTION_IDENTIFIER == 22 // regular multiplication
        a[i] = a[i].prepare_mult(b[i]);
#elif FUNCTION_IDENTIFIER == 23 // fixed point multiplication
        a[i] = a[i].prepare_dot(b[i]);
        a[i].mask_and_send_dot();
#endif
    }
    Share::communicate();
    for(int i = 0; i < DATTYPE; i++)
    {
#if FUNCTION_IDENTIFIER == 22 // regular multiplication
        a[i].complete_mult_without_trunc();
#elif FUNCTION_IDENTIFIER == 23 // fixed point multiplication
        a[i].complete_mult();
#endif
    }
    Share::communicate();
}
a[0].prepare_reveal_to_all(); // dummy reveal
Share::communicate();
*res = a[0].complete_reveal_to_all();
}

#elif FUNCTION_IDENTIFIER == 24

template<typename Share>
void A2Bit_Setup_Round_Test(DATATYPE* res)
{
using A = Additive_Share<DATATYPE, Share>;
const int sequential_rounds = NUM_INPUTS; //specify number of rounds 
using S = XOR_Share<DATATYPE, Share>;
using Bitset = sbitset_t<BITLENGTH, S>;
using sint = sint_t<A>;

Share::communicate(); // dummy round
const int len = 1;
Bitset *s1 = new Bitset[len];
Bitset *s2 = new Bitset[len];
sint *val = new sint[len];
for(int q = 0; q < sequential_rounds; q++)
{
        for(int i = 0; i < len; i++)
        {
            s1[i] = Bitset::prepare_A2B_S1((S*) val[i].get_share_pointer());
            s2[i] = Bitset::prepare_A2B_S2((S*) val[i].get_share_pointer());
        }
        Share::communicate();
        for(int i = 0; i < len; i++)
        {
            s1[i].complete_A2B_S1();
            s2[i].complete_A2B_S2();
        }
        Share::communicate();
}
A dummy;
dummy.prepare_reveal_to_all(); // dummy reveal
Share::communicate();
*res = dummy.complete_reveal_to_all();
delete[] s1;
delete[] s2;
delete[] val;
}

#elif FUNCTION_IDENTIFIER == 25
template<typename Share>
void BIT2A_Setup_Round_Test(DATATYPE* res)
{
using A = Additive_Share<DATATYPE, Share>;
const int sequential_rounds = NUM_INPUTS; //specify number of rounds 
using S = XOR_Share<DATATYPE, Share>;
using Bitset = sbitset_t<BITLENGTH, S>;
using sint = sint_t<A>;
S* y = new S;
sint* t1 = new sint;
sint* t2 = new sint;
    for(int j = 0; j < sequential_rounds; j++)
    {
        for(int i = 0; i < 1; i++)
        {
            y[i].prepare_bit_injection_S1(t1[i].get_share_pointer());
            y[i].prepare_bit_injection_S2(t2[i].get_share_pointer());
        }
        Share::communicate();
        for(int i = 0; i < 1; i++)
        {
            t1[i].complete_bit_injection_S1();
            t2[i].complete_bit_injection_S2();
        }
    }
delete y;
delete t1;
delete t2;
A dummy;
dummy.prepare_reveal_to_all();
Share::communicate();
*res = dummy.complete_reveal_to_all();
}

#elif FUNCTION_IDENTIFIER == 26
    
    template<typename Share>
void dot_prod_round_bench(DATATYPE* res)
{
    using M = Additive_Share<DATATYPE, Share>;
    Share::communicate(); // dummy round
    const int sequential_rounds = NUM_INPUTS; // low number of rounds due to high computational complexity
    const int dot_prod_size = 20000;
    auto a = new M[dot_prod_size];
    auto b = new M[dot_prod_size][dot_prod_size];
    auto c = new M[dot_prod_size];
  for(int rounds = 0; rounds < sequential_rounds; rounds++)
  {
    for(int i = 0; i < dot_prod_size; i++)
    {
        for(int j = 0; j < dot_prod_size; j++)
        {
            c[i] += a[i].prepare_dot(b[j][i]);
        }
        c[i].mask_and_send_dot_without_trunc();
    }
    Share::communicate();
    for(int i = 0; i < dot_prod_size; i++)
    {
            c[i].complete_mult_without_trunc();
    }
    Share::communicate();
  }
    c[dot_prod_size-1].prepare_reveal_to_all();
    Share::communicate();
    *res = c[dot_prod_size-1].complete_reveal_to_all();

delete[] a;
delete[] b;
delete[] c;
}


#endif


template<typename Share>
void AND_bench(DATATYPE* res)
{
    using S = XOR_Share<DATATYPE, Share>;
    auto a = new S[NUM_INPUTS];
    auto b = new S[NUM_INPUTS];
    auto c = new S[NUM_INPUTS];
    Share::communicate(); // dummy round
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        c[i] = a[i] & b[i];
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        c[i].complete_and();
    }
    Share::communicate();

    c[0].prepare_reveal_to_all();

    Share::communicate();

    *res = c[0].complete_reveal_to_all();

}

template<typename Share, typename Datatype>
static void trunc_2k_in_place(sint_t<Additive_Share<Datatype, Share>>*  val, const int len){
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;
    for(int i = 0; i < len; i++)
        val[i].prepare_reveal_to_all();
    Share::communicate();
    UINT_TYPE dummy[DATTYPE];
    for(int i = 0; i < len; i++)
    {
        val[i].complete_reveal_to_all(dummy);
        std::cout << "val: " << dummy[0] << std::endl;
    }
    
    sint* r_msb = new sint[len];
    sint* r_mk2 = new sint[len];
    sint* c = new sint[len];
    sint* c_prime = new sint[len];
    sint* b = new sint[len];
    for(int i = 0; i < len; i++)
    {
        val[i].prepare_trunc_2k_inputs(r_mk2[i], r_msb[i], c[i], c_prime[i]);
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        val[i].complete_trunc_2k_inputs(r_mk2[i], r_msb[i],c[i], c_prime[i]);
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        b[i].prepare_XOR(r_msb[i],c[i]);
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        b[i].complete_XOR(r_msb[i],c[i]);
    } 
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        b[i] = b[i].mult_public(UINT_TYPE(1) << (BITLENGTH - FRACTIONAL - 1));
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        val[i] = c_prime[i] + b[i] - r_mk2[i];
    }
    for(int i = 0; i < len; i++)
        c_prime[i].prepare_reveal_to_all();
    for(int i = 0; i < len; i++)
        b[i].prepare_reveal_to_all();
    for(int i = 0; i < len; i++)
        r_mk2[i].prepare_reveal_to_all();
    for(int i = 0; i < len; i++)
        c[i].prepare_reveal_to_all();
    for(int i = 0; i < len; i++)
        r_msb[i].prepare_reveal_to_all();
   
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        c_prime[i].complete_reveal_to_all(dummy);
        std::cout << "c_prime: " << dummy[0] << std::endl;
    }
    for(int i = 0; i < len; i++)
    {
        b[i].complete_reveal_to_all(dummy);
        std::cout << "b: " << dummy[0] << std::endl;
    }
    for(int i = 0; i < len; i++)
    {
        r_mk2[i].complete_reveal_to_all(dummy);
        std::cout << "r_mk2: " << dummy[0] << std::endl;
    }
    for(int i = 0; i < len; i++)
    {
        c[i].complete_reveal_to_all(dummy);
        std::cout << "c: " << dummy[0] << std::endl;
    }
    for(int i = 0; i < len; i++)
    {
        r_msb[i].complete_reveal_to_all(dummy);
        std::cout << "r_msb: " << dummy[0] << std::endl;
    }
    delete[] r_mk2;
    delete[] r_msb;
    delete[] c_prime;
    delete[] c;
    delete[] b;
}

template<typename Share, typename Datatype>
void testo(sint_t<Additive_Share<Datatype, Share>>*  test, const int len)
{
    /* using M = Matrix_Share<DATATYPE, Share>; */
    using M = Additive_Share<DATATYPE, Share>;
    /* using M = sint_t<Additive_Share<DATATYPE, Share>>; */
    using sint = sint_t<M>;
    for(int i = 0; i < len; i++)
    {
        test[i].template prepare_receive_and_replicate<P_0>(400);
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        test[i].template complete_receive_from<P_0>();
        test[i].prepare_reveal_to_all();
    }
    Share::communicate();
    UINT_TYPE dummy[DATTYPE];
    for(int i = 0; i < len; i++)
    {
        test[i].complete_reveal_to_all(dummy);
        std::cout << "test: " << dummy[0] << std::endl;
    }

}

    template<typename Share>
void fixed_test(DATATYPE* res)
{
    /* using M = Matrix_Share<DATATYPE, Share>; */
    using M = Additive_Share<DATATYPE, Share>;
    /* using M = sint_t<Additive_Share<DATATYPE, Share>>; */
    using sint = sint_t<M>;


    auto test = new sint[NUM_INPUTS][NUM_INPUTS];
    testo( (sint*) test ,NUM_INPUTS*NUM_INPUTS);


    auto a = new sint[NUM_INPUTS][NUM_INPUTS];
    auto b = new sint[NUM_INPUTS][NUM_INPUTS];
    auto c = new sint[NUM_INPUTS][NUM_INPUTS];
    /* std::memset(c, 0, sizeof(M) * NUM_INPUTS * NUM_INPUTS); */
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            a[i][j]. template prepare_receive_from<P_0>();
            b[i][j]. template prepare_receive_from<P_2>();
        }
}
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            a[i][j]. template complete_receive_from<P_0>();
            b[i][j]. template complete_receive_from<P_2>();
        }
    }


    Share::communicate(); // dummy round
    /* for(int i = 0; i < NUM_INPUTS; i++) */
    /* { */
    /*     for(int j = 0; j < NUM_INPUTS; j++) */
    /*     { */
    /*         for(int k = 0; k < NUM_INPUTS; k++) */
    /*         { */
    /*             /1* c[i][j] = c[i][j] + a[i][k] * b[k][j]; *1/ */
    /*             c[i][j] = c[i][j] + a[i][k].prepare_dot(b[k][j]); */
    /*             /1* c[i][j].prepare_dot_add(a[i][k], b[k][j]); *1/ */
    /*         } */
    /*             c[i][j].mask_and_send_dot_without_trunc(); */
    /*             /1* c[i][j].mask_and_send_dot(); *1/ */
    /*     } */
    /* } */
    /* Share::communicate(); */
    /* for(int i = 0; i < NUM_INPUTS; i++) */
    /* { */
    /*     for(int j = 0; j < NUM_INPUTS; j++) */
    /*     { */
    /*         c[i][j].complete_mult_without_trunc(); */
    /*         /1* c[i][j].complete_mult(); *1/ */
    /*     } */
/* } */
    /* Share::communicate(); */
/* delete[] a; */
/* delete[] b; */
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
        c[i][j] = a[i][j] + b[i][j];
        }   
    }
    trunc_2k_in_place( (sint*) c,NUM_INPUTS*NUM_INPUTS);


    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            c[i][j].prepare_reveal_to_all();
        }
    }
    auto result_arr = new UINT_TYPE[NUM_INPUTS*NUM_INPUTS][DATTYPE];
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            c[i][j].complete_reveal_to_all(result_arr[NUM_INPUTS*i+j]);
            }
        }
            if(current_phase == PHASE_LIVE)
    {
        std::cout << "P" << PARTY << ": Result: ";
        for (int i = 0; i < NUM_INPUTS*NUM_INPUTS; i++)
        {
    for(int j = 0; j < DATTYPE; j++)
    {
#if FRACTIONAL > 0
        std::cout << fixedToFloat<float, UINT_TYPE, FRACTIONAL>(result_arr[i][j]) << " ";
#else
        std::cout << result_arr[i][j] << " ";
#endif
    std::cout << std::endl;
    }
    std::cout << std::endl;
}

    }



delete[] c;
delete[] result_arr;


}



    template<typename Share>
void dot_prod_bench(DATATYPE* res)
{
    Share::communicate(); // dummy round
    /* using M = Matrix_Share<DATATYPE, Share>; */
    using M = Additive_Share<DATATYPE, Share>;
    auto a = new M[NUM_INPUTS];
    auto b = new M[NUM_INPUTS][NUM_INPUTS];
    auto c = new M[NUM_INPUTS];
    Share::communicate(); // dummy round
    for(int i = 0; i < NUM_INPUTS; i++)
    {
#if FUNCTION_IDENTIFIER == 14
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            c[i] += a[i].prepare_dot(b[i][j]);
        }
#endif
        c[i].mask_and_send_dot_without_trunc();
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
            c[i].complete_mult_without_trunc();
    }

    Share::communicate();
    c[NUM_INPUTS-1].prepare_reveal_to_all();
    Share::communicate();
    *res = c[NUM_INPUTS-1].complete_reveal_to_all();

delete[] a;
delete[] b;
delete[] c;

}



    template<typename Share>
void RELU_bench(DATATYPE* res)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    using sint = sint_t<A>;
    
    sint* val = new sint[NUM_INPUTS];
    Bitset *s1 = new Bitset[NUM_INPUTS];
    Bitset *s2 = new Bitset[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        s1[i] = Bitset::prepare_A2B_S1( (S*) val[i].get_share_pointer());
        s2[i] = Bitset::prepare_A2B_S2( (S*) val[i].get_share_pointer());
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        s1[i].complete_A2B_S1();
        s2[i].complete_A2B_S2();
    }
    /* Bitset* y = new Bitset[NUM_INPUTS]; */
    S *y = new S[NUM_INPUTS];
    /* BooleanAdder<S> *adder = new BooleanAdder<S>[NUM_INPUTS]; */
    #if FUNCTION_IDENTIFIER == 16
    std::vector<PPA_MSB_Unsafe<BITLENGTH,S>> adders;
    #else
    std::vector<BooleanAdder_MSB<BITLENGTH,S>> adders;
    #endif
    adders.reserve(NUM_INPUTS);
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], y[i]);
    }
    while(!adders[0].is_done())
    {
        for(int i = 0; i < NUM_INPUTS; i++)
        {
            adders[i].step();
        }
        Share::communicate();
    }
    delete[] s1;
    delete[] s2;
    adders.clear();
    adders.shrink_to_fit();
    
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        y[i] = ~ y[i];
    }
    sint* t1 = new sint[NUM_INPUTS];
    sint* t2 = new sint[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        y[i].prepare_bit_injection_S1(t1[i].get_share_pointer());
        y[i].prepare_bit_injection_S2(t2[i].get_share_pointer());
    }
    delete[] y;
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        t1[i].complete_bit_injection_S1();
        t2[i].complete_bit_injection_S2();
    }
    sint* result = new sint[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        result[i].prepare_XOR(t1[i],t2[i]);
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        result[i].complete_XOR(t1[i],t2[i]);
    }
    delete[] t1;
    delete[] t2;

    for(int i = 0; i < NUM_INPUTS; i++)
    {
        result[i] = result[i].prepare_mult(val[i]);
    }
    delete[] val;
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        result[i].complete_mult_without_trunc();
    }

    // dummy reveal for sync
    result[NUM_INPUTS-1].prepare_reveal_to_all();
    Share::communicate();
    UINT_TYPE dummy[DATTYPE];
    result[NUM_INPUTS-1].complete_reveal_to_all(dummy);


    


}

#if FUNCTION_IDENTIFIER == 28

    /* template<typename Share, typename Datatype> */
/* XOR_Share<Datatype, Share> bitext(sint_t<Additive_Share<Datatype, Share>> x,sint_t<Additive_Share<Datatype, Share>> y) */
/* { */
/* using S = XOR_Share<Datatype, Share>; */
/* using A = Additive_Share<Datatype, Share>; */
/* using Bitset = sbitset_t<BITLENGTH,S>; */
/* using sint = sint_t<A>; */
    /* sint* val = new sint; */
    /* Bitset *s1 = new Bitset; */
    /* Bitset *s2 = new Bitset; */
    /* for(int i = 0; i < NUM_INPUTS; i++) */
    /* { */
    /*     s1[i] = sbitset_t<BITLENGTH,S>::prepare_A2B_S1( (S*) val[i].get_share_pointer()); */
    /*     s2[i] = sbitset_t<BITLENGTH,S>::prepare_A2B_S2( (S*) val[i].get_share_pointer()); */
    /* } */
    /* Share::communicate(); */
    /* for(int i = 0; i < NUM_INPUTS; i++) */
    /* { */
    /*     s1[i].complete_A2B_S1(); */
    /*     s2[i].complete_A2B_S2(); */
    /* } */
    /* /1* Bitset* y = new Bitset[NUM_INPUTS]; *1/ */
    /* S *y = new S[NUM_INPUTS]; */
    /* /1* BooleanAdder<S> *adder = new BooleanAdder<S>[NUM_INPUTS]; *1/ */
    /* #if FUNCTION_IDENTIFIER == 16 */
    /* std::vector<PPA_MSB_Unsafe<S>> adders; */
    /* #else */
    /* std::vector<BooleanAdder_MSB<S>> adders; */
    /* #endif */
    /* adders.reserve(NUM_INPUTS); */
    /* for(int i = 0; i < NUM_INPUTS; i++) */
    /* { */
    /*     /1* adder[i].set_values(s1[i], s2[i], y[i]); *1/ */
    /*     adders.emplace_back(s1[i], s2[i], y[i]); */
    /* } */
    /* while(!adders[0].is_done()) */
    /* { */
    /*     for(int i = 0; i < NUM_INPUTS; i++) */
    /*     { */
    /*         adders[i].step(); */
    /*     } */
    /*     Share::communicate(); */
    /* } */
    /* delete[] s1; */
    /* delete[] s2; */
    /* adders.clear(); */
    /* adders.shrink_to_fit(); */
    
    /* for(int i = 0; i < NUM_INPUTS; i++) */
    /* { */
    /*     y[i] = ~ y[i]; */
    /* } */


/* } */


    /* template<typename Share> */
/* Share argmax_helper(int begin, int end, Share* x, Share* d) */
/* { */
/* int m = end - begin; */
/* if(m == 1) */
/* { */
    /* d[begin] = SET_ALL_ONE(); */
    /* auto y = x[begin]; */
    /* return y; */
/* } */
/* else if(m == 2) */
/* { */
    /* d[begin] = bitext(x[begin], x[end]); */
    /* d[end] = d[begin] ^ SET_ALL_ONE(); */
    /* auto y = obv(x[end], x[begin], d[begin]); */
    /* return y; */
/* } */
/* else if(m == 3) */
/* { */
    /* auto d1 = bitext(x[begin], x[begin+1]); */
    /* auto y1 = obv(x[begin+1], x[begin], d1); */
    /* auto d2 = bitext(y1, x[end]); */
    /* auto y = obv(x[end], y1, d2); */
    /* d[begin] = d1 & d2; */
    /* d[begin+1] = d2 ^ d[begin]; */
    /* d[end] = ! (d1 ^ d2); */ 
    /* return y; */
/* } */
/* auto y1 = argmax_helper(begin, begin + m/2, x, d); */
/* auto y2 = argmax_helper(begin + m/2 + 1, end, x, d); */
/* auto db = bitext(y1, y2); */
/* auto y = obv(y2, y1, db); */
/* for (int i = begin; i < begin + m/2; i++) */
/* { */
    /* d[i] = d[i] & db; */
/* } */
/* for (int i = begin + m/2 + 1; i < end; i++) */
/* { */
    /* d[i] = d[i] & !db; */
/* } */
/* return y; */
/* } */

    /* template<typename Share> */
/* void argmax(Share* begin, Share* end, Share* output) */
/* { */
    /* argmax_helper(0, end - begin, begin, output); */
/* } */

// Promote bit to arithmetic sharing
template<typename Share, typename Datatype>
void bitinj_range(XOR_Share<Datatype, Share>* bit_val, int len, sint_t<Additive_Share<Datatype, Share>>* output)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using sint = sint_t<A>;
sint* t1 = new sint[len];
sint* t2 = new sint[len];
for (int i = 0; i < len; i++)
{
    bit_val[i].prepare_bit_injection_S1(t1[i].get_share_pointer());
    bit_val[i].prepare_bit_injection_S2(t2[i].get_share_pointer());
}
Share::communicate();
for (int i = 0; i < len; i++)
{
    t1[i].complete_bit_injection_S1();
    t2[i].complete_bit_injection_S2();
}
for (int i = 0; i < len; i++)
{
    output[i].prepare_XOR(t1[i], t2[i]);
}
Share::communicate();
for (int i = 0; i < len; i++)
{
    output[i].complete_XOR(t1[i], t2[i]);
}
delete[] t1;
delete[] t2;

}

// compute msbs of a range of arithemtic shares
template<int k, typename Datatype, typename Share>
void get_msb_range(sint_t<Additive_Share<Datatype, Share>>* val, XOR_Share<Datatype, Share>* msb, int len)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using Bitset = sbitset_t<k,S>;
using sint = sint_t<A>;
    Bitset *s1 = new Bitset[len];
Bitset *s2 = new Bitset[len];
    for(int i = 0; i < len; i++)
    {
        s1[i] = Bitset::prepare_A2B_S1( (S*) val[i].get_share_pointer());
        s2[i] = Bitset::prepare_A2B_S2( (S*) val[i].get_share_pointer());
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        s1[i].complete_A2B_S1();
        s2[i].complete_A2B_S2();
    }
    /* Bitset* y = new Bitset[NUM_INPUTS]; */

    /* std::vector<BooleanAdder_MSB<S>> adders; */
    /* std::vector<PPA_MSB_Unsafe<S>> adders; */
    /* std::vector<PPA_MSB_4Way<Bitset::get_bitlength(), S> > adders; */
    std::vector<BooleanAdder_MSB<Bitset::get_bitlength(), S> > adders;
    
    adders.reserve(len);
    for(int i = 0; i < len; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], msb[i]);
    }
    while(!adders[0].is_done())
    {
        for(int i = 0; i < len; i++)
        {
            adders[i].step();
        }
        Share::communicate();
    }
    delete[] s1;
    delete[] s2;
    adders.clear();
    adders.shrink_to_fit();
    

}

template<int k, typename Datatype, typename Share>
void max_min_msb_range(sint_t<Additive_Share<Datatype, Share>>* val, XOR_Share<Datatype, Share>* msb, int m, bool want_max)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using sint = sint_t<A>;

       sint* max_val = new sint[(m+1)/2];
       int offset = m % 2; // if m is odd, offset is 1
       int counter = 0;
       for(int j = 1; j < m; j+=2)
       {
           if(want_max)
        max_val[counter] = val[j] - val[j-1];
           else
        max_val[counter] = val[j-1] - val[j];
            counter++;
       }
            /* #if PARTY == 2 */
        /* for(int i = 0; i < counter; i++) */
            /* std::cout << "val: " << val[i].get_p1() << std::endl; */
        /* #endif */ 

            /* #if PARTY == 2 */
        /* for(int i = 0; i < counter; i++) */
            /* std::cout << "max val: " << max_val[i].get_p1() << std::endl; */
        /* #endif */ 

    get_msb_range<k>(max_val, msb, counter);
            /* #if PARTY == 2 */
        /* for(int i = 0; i < counter; i++) */
            /* std::cout << "msb: " << msb[i].get_p1() << std::endl; */
        /* #endif */ 

    delete[] max_val;

    // get arithmetic version of msb to update values
    sint* max_idx = new sint[counter];
    bitinj_range(msb, counter, max_idx);

    for(int i = 0; i < counter; i++)
    {
/* #if PARTY ==2 */
/*             std::cout << "max idx: " << max_idx[i].get_p1() << std::endl; */
/* #endif */
        max_idx[i] = max_idx[i] * (val[2*i] - val[2*i+1]);
    }
    Share::communicate();
    for(int i = 0; i < counter; i++)
    {
        max_idx[i].complete_mult();
        max_idx[i] = max_idx[i] + val[2*i+1];
        val[i] = max_idx[i];
            /* #if PARTY == 2 */
            /* std::cout << "updated val: " << val[i].get_p1() << std::endl; */
/* #endif */

    }
       if(offset == 1)
       {
        val[counter] = val[m-1]; // last uneven element is always pairwise max
        msb[counter] = SET_ALL_ONE();
       }
    delete[] max_idx;

}
    
    template<int k, typename Datatype, typename Share>
sint_t<Additive_Share<Datatype, Share>> max_min(sint_t<Additive_Share<Datatype, Share>>* begin, sint_t<Additive_Share<Datatype, Share>>* end, bool want_max)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using sint = sint_t<A>;
   int m = end - begin;
   int og_len = m;

   sint* val = new sint[m];
    std::copy(begin, end, val);
   if(m == 1)
   {
       sint ret = val[0];
       delete[] val;
       return ret;
   }

   int log2m = std::ceil(std::log2(m)); 
   for(int i = 0; i < log2m; i++)
   {
       int counter = m/2; // 
        int offset = m % 2; // if m is odd, offset is 1
        S* msb = new S[counter+offset];
        if(want_max)
            max_min_msb_range<k>(val,msb,m,true); //get msb and max of 0 -> counter
        else
            max_min_msb_range<k>(val,msb,m,false); //get msb and max of 0 -> counter

        delete[] msb;
        m = counter + offset;
    }
   sint ret = val[0];
   delete[] val;
   return ret;
}


    template<int c, typename Datatype, typename Share>
void argmax_argmin(sint_t<Additive_Share<Datatype, Share>>* begin, sint_t<Additive_Share<Datatype, Share>>* end, XOR_Share<Datatype, Share>* output, bool want_max)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using sint = sint_t<A>;
   int m = end - begin;
   int og_len = m;
   if(m == 1)
   {
       output[0] = SET_ALL_ONE();
       return;
   }

   sint* val = new sint[m];
    std::copy(begin, end, val);
                /* #if PARTY == 2 */
        /* for(int i = 0; i < m; i++) */
            /* std::cout << "val: " << val[i].get_p1() << std::endl; */
        /* #endif */ 

   int log2m = std::ceil(std::log2(m)); 
   for(int i = 0; i < log2m; i++)
   {
       int counter = m/2; // 
        int offset = m % 2; // if m is odd, offset is 1
        S* msb = new S[m];
        if(want_max)
            max_min_msb_range<c>(val,msb,m,true); //get msb and max of 0 -> counter
        else
            max_min_msb_range<c>(val,msb,m,false); //get msb and max of 0 -> counter

        //update args
       if (i == 0) // first round
       {
            for(int j = 1; j < m; j+=2)
            {
                output[j-1] = msb[j/2];
                output[j] = !msb[j/2];
            }
            if(offset == 1)
            {
                output[m-1] = SET_ALL_ONE(); // single element is always max
            }
       }
       else
       {
           int jump = 1 << (i+1); // i = 1 -> jump = 4, 4 values are being compared in total
            for(int j = 0; j < counter; j++)
            {
                for(int k = 0; k < jump && j*jump+k < og_len ; k++)
                {
                    if(k < jump/2)
                    {
                        output[j*jump+k] = output[j*jump+k] & msb[j];
                    }
                    else
                    {
                        output[j*jump+k] = output[j*jump+k] & !msb[j];
                    }
                }
            }
            Share::communicate();
            for(int j = 0; j < counter; j++)
            {
                for(int k = 0; k < jump && j*jump+k < og_len ; k++)
                {
                        output[j*jump+k].complete_and();
                }
            }
       }
        delete[] msb;
        m = counter + offset;
       }
}

template<typename Share>
void argmax_test(DATATYPE* res)
{
using S = XOR_Share<DATATYPE, Share>;
using A = Additive_Share<DATATYPE, Share>;
using sint = sint_t<A>;
/* const int k = REDUCED_BITLENGTH; */
const int k = BITLENGTH;
auto a = new sint[NUM_INPUTS];
auto max_output = new S[NUM_INPUTS];
auto min_output = new S[NUM_INPUTS];
for(int i = 0; i < NUM_INPUTS; i++)
        a[i]. template prepare_receive_from<P_2>();
Share::communicate();
for(int i = 0; i < NUM_INPUTS; i++)
        a[i]. template complete_receive_from<P_2>();
Share::communicate();
                /* #if PARTY == 2 */
        /* for(int i = 0; i < NUM_INPUTS; i++) */
            /* std::cout << "a: " << a[i].get_p1() << std::endl; */
        /* #endif */ 
argmax_argmin<k>(a, a+NUM_INPUTS, max_output,true);
argmax_argmin<k>(a, a+NUM_INPUTS, min_output,false);
auto max_val = max_min<k>(a, a+NUM_INPUTS, true);
auto min_val = max_min<k>(a, a+NUM_INPUTS, false);
for(int i = 0; i < NUM_INPUTS; i++)
{
        max_output[i].prepare_reveal_to_all();
        min_output[i].prepare_reveal_to_all();
}
        max_val.prepare_reveal_to_all();
        min_val.prepare_reveal_to_all();
Share::communicate();
auto result_arr = new DATATYPE[2][NUM_INPUTS];
for(int i = 0; i < NUM_INPUTS; i++)
{
        result_arr[0][i] = max_output[i].complete_reveal_to_all();
        result_arr[1][i] = min_output[i].complete_reveal_to_all();
}
auto max_int = NEW( UINT_TYPE[BITLENGTH * sizeof(DATATYPE)/sizeof(UINT_TYPE)]);
auto min_int = NEW( UINT_TYPE[BITLENGTH * sizeof(DATATYPE)/sizeof(UINT_TYPE)]);
max_val.complete_reveal_to_all(max_int);
min_val.complete_reveal_to_all(min_int);
if(current_phase == PHASE_LIVE)
{
#if DATTYPE <= 64
for(int i = 0; i < NUM_INPUTS; i++)
    std::cout << "arg_max: " << "Index: " << i << " Value: " << result_arr[0][i] << std::endl;
for(int i = 0; i < NUM_INPUTS; i++)
    std::cout << "arg_min: " << "Index: " << i << " Value: "<< result_arr[1][i] << std::endl;
#endif
std::cout << "max: " << max_int[0] << std::endl;
std::cout << "min: " << min_int[0] << std::endl;
}
delete[] a;
delete[] max_output;
delete[] min_output;
delete[] result_arr;
}


#endif





#if FUNCTION_IDENTIFIER == 29 || FUNCTION_IDENTIFIER == 30
template<typename Share>
void mult34_test(DATATYPE* res)
{
using A = Additive_Share<DATATYPE, Share>;
A* inputs = new A[NUM_INPUTS];
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            inputs[j]. template prepare_receive_from<P_0>();
        }
    Share::communicate();
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            inputs[j]. template complete_receive_from<P_0>();
        }
#if FUNCTION_IDENTIFIER == 29 //mult3 
A result1 = inputs[0].prepare_mult3(inputs[1], inputs[2]);
A result2 = inputs[0].prepare_mult3(inputs[1], inputs[2]);
A result3 = inputs[0].prepare_mult3(inputs[1], inputs[2]);
A ver_result = inputs[0] * inputs[1];
Share::communicate();
result1.complete_mult3();
result2.complete_mult3();
result3.complete_mult3();
ver_result.complete_mult();

A result_tot = result1.prepare_mult3(result2, result3);
ver_result = ver_result * inputs[2];
Share::communicate();
result_tot.complete_mult3();
ver_result.complete_mult();
A ver_result_tot = ver_result * ver_result;
Share::communicate();
ver_result_tot.complete_mult();
ver_result_tot = ver_result_tot * ver_result;
Share::communicate();
ver_result_tot.complete_mult();
result1.prepare_reveal_to_all();
ver_result.prepare_reveal_to_all();
result_tot.prepare_reveal_to_all();
ver_result_tot.prepare_reveal_to_all();
Share::communicate();
DATATYPE* result_arr = new DATATYPE[4];
result_arr[0] = result1.complete_reveal_to_all();
result_arr[1]= ver_result.complete_reveal_to_all();
result_arr[2] = result_tot.complete_reveal_to_all();
result_arr[3]= ver_result_tot.complete_reveal_to_all();
#elif FUNCTION_IDENTIFIER == 30 // mult4 
A result1 = inputs[0].prepare_mult4(inputs[1], inputs[2], inputs[3]);
A result2 = inputs[0].prepare_mult4(inputs[1], inputs[2], inputs[3]);
A result3 = inputs[0].prepare_mult4(inputs[1], inputs[2], inputs[3]);
A result4 = inputs[0].prepare_mult4(inputs[1], inputs[2], inputs[3]);
A ver_result = inputs[0] * inputs[1];
Share::communicate();
result1.complete_mult4();
result2.complete_mult4();
result3.complete_mult4();
result4.complete_mult4();
ver_result.complete_mult();
A result_tot = result1.prepare_mult4(result2, result3, result4);
ver_result = ver_result * inputs[2];
Share::communicate();
result_tot.complete_mult4();
ver_result.complete_mult();
ver_result = ver_result * inputs[3];
Share::communicate();
ver_result.complete_mult();
A ver_result_tot = ver_result * ver_result;
Share::communicate();
ver_result_tot.complete_mult();
ver_result_tot = ver_result_tot * ver_result;
Share::communicate();
ver_result_tot.complete_mult();
ver_result_tot = ver_result_tot * ver_result;
Share::communicate();
ver_result_tot.complete_mult();
result1.prepare_reveal_to_all();
ver_result.prepare_reveal_to_all();
result_tot.prepare_reveal_to_all();
ver_result_tot.prepare_reveal_to_all();
Share::communicate();
DATATYPE* result_arr = new DATATYPE[4];
result_arr[0] = result1.complete_reveal_to_all();
result_arr[1]= ver_result.complete_reveal_to_all();
result_arr[2] = result_tot.complete_reveal_to_all();
result_arr[3]= ver_result_tot.complete_reveal_to_all();
#endif
if(current_phase == PHASE_LIVE)
{
    std::cout << "P" << PARTY << " result: " << std::to_string(result_arr[0]) << " ver_result: " << std::to_string(result_arr[1]) << std::endl;
    std::cout << "P" << PARTY <<  " result2: " << std::to_string(result_arr[2]) << " ver_result2: " << std::to_string(result_arr[3]) << std::endl;
}


}


#endif

#if FUNCTION_IDENTIFIER == 31 || FUNCTION_IDENTIFIER == 32 || FUNCTION_IDENTIFIER == 33 || FUNCTION_IDENTIFIER == 34
template<typename Share>
void dot234_test(DATATYPE* res)
{
using A = Additive_Share<DATATYPE, Share>;
A* inputs = new A[NUM_INPUTS];
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            inputs[j]. template prepare_receive_from<P_0>();
        }
    Share::communicate();
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            inputs[j]. template complete_receive_from<P_0>();
        }
#if FUNCTION_IDENTIFIER == 31 //dot2
A result1 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot(inputs[3]) + inputs[4].prepare_dot(inputs[5]) + inputs[6].prepare_dot(inputs[7]);
A result2 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot(inputs[3]) + inputs[4].prepare_dot(inputs[5]) + inputs[6].prepare_dot(inputs[7]);
A result3 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot(inputs[3]) + inputs[4].prepare_dot(inputs[5]) + inputs[6].prepare_dot(inputs[7]);
A result4 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot(inputs[3]) + inputs[4].prepare_dot(inputs[5]) + inputs[6].prepare_dot(inputs[7]);
result1.mask_and_send_dot_without_trunc();
result2.mask_and_send_dot_without_trunc();
result3.mask_and_send_dot_without_trunc();
result4.mask_and_send_dot_without_trunc();
Share::communicate();
result1.complete_mult_without_trunc();
result2.complete_mult_without_trunc();
result3.complete_mult_without_trunc();
result4.complete_mult_without_trunc();
result1 = result1.prepare_dot(result2) + result3.prepare_dot(result4);
result1.mask_and_send_dot_without_trunc();
Share::communicate();
result1.complete_mult_without_trunc();
result1.prepare_reveal_to_all();
Share::communicate();
DATATYPE* result_arr = new DATATYPE;
result_arr[0] = result1.complete_reveal_to_all();
#elif FUNCTION_IDENTIFIER == 32 // dot3
A result1 = inputs[0].prepare_dot3(inputs[1],inputs[2]) + inputs[3].prepare_dot3(inputs[4],inputs[5]) + inputs[6].prepare_dot3(inputs[7],inputs[8]) + inputs[9].prepare_dot3(inputs[10],inputs[11]);
A result2 = inputs[0].prepare_dot3(inputs[1],inputs[2]) + inputs[3].prepare_dot3(inputs[4],inputs[5]) + inputs[6].prepare_dot3(inputs[7],inputs[8]) + inputs[9].prepare_dot3(inputs[10],inputs[11]);
A result3 = inputs[0].prepare_dot3(inputs[1],inputs[2]) + inputs[3].prepare_dot3(inputs[4],inputs[5]) + inputs[6].prepare_dot3(inputs[7],inputs[8]) + inputs[9].prepare_dot3(inputs[10],inputs[11]);
A result4 = inputs[0].prepare_dot3(inputs[1],inputs[2]) + inputs[3].prepare_dot3(inputs[4],inputs[5]) + inputs[6].prepare_dot3(inputs[7],inputs[8]) + inputs[9].prepare_dot3(inputs[10],inputs[11]);
A result5 = inputs[0].prepare_dot3(inputs[1],inputs[2]) + inputs[3].prepare_dot3(inputs[4],inputs[5]) + inputs[6].prepare_dot3(inputs[7],inputs[8]) + inputs[9].prepare_dot3(inputs[10],inputs[11]);
A result6 = inputs[0].prepare_dot3(inputs[1],inputs[2]) + inputs[3].prepare_dot3(inputs[4],inputs[5]) + inputs[6].prepare_dot3(inputs[7],inputs[8]) + inputs[9].prepare_dot3(inputs[10],inputs[11]);
result1.mask_and_send_dot_without_trunc();
result2.mask_and_send_dot_without_trunc();
result3.mask_and_send_dot_without_trunc();
result4.mask_and_send_dot_without_trunc();
result5.mask_and_send_dot_without_trunc();
result6.mask_and_send_dot_without_trunc();
Share::communicate();
result1.complete_mult_without_trunc();
result2.complete_mult_without_trunc();
result3.complete_mult_without_trunc();
result4.complete_mult_without_trunc();
result5.complete_mult_without_trunc();
result6.complete_mult_without_trunc();
/* result1 = result1.prepare_dot3(result2,result3) + result4.prepare_dot3(result5,result6); */
result1 = result1.prepare_dot(result2 + result3) - (inputs[0] + inputs[3]).prepare_dot3(inputs[1],inputs[2]);
        //val[i] = val[i].prepare_dot( t1[i] + t2[i]) - (val[i] + val[i]).prepare_dot3(t1[i],t2[i]); // (a+b) v - 2abv
result1.mask_and_send_dot();
Share::communicate();
result1.complete_mult();
result1.prepare_reveal_to_all();
Share::communicate();
DATATYPE* result_arr = new DATATYPE;
result_arr[0] = result1.complete_reveal_to_all();
#elif FUNCTION_IDENTIFIER == 33 // dot4
A result1 = inputs[0].prepare_dot4(inputs[1],inputs[2],inputs[3]) + inputs[4].prepare_dot4(inputs[5],inputs[6],inputs[7]) + inputs[8].prepare_dot4(inputs[9],inputs[10],inputs[11]) + inputs[12].prepare_dot4(inputs[13],inputs[14],inputs[15]);
A result2 = inputs[0].prepare_dot4(inputs[1],inputs[2],inputs[3]) + inputs[4].prepare_dot4(inputs[5],inputs[6],inputs[7]) + inputs[8].prepare_dot4(inputs[9],inputs[10],inputs[11]) + inputs[12].prepare_dot4(inputs[13],inputs[14],inputs[15]);
A result3 = inputs[0].prepare_dot4(inputs[1],inputs[2],inputs[3]) + inputs[4].prepare_dot4(inputs[5],inputs[6],inputs[7]) + inputs[8].prepare_dot4(inputs[9],inputs[10],inputs[11]) + inputs[12].prepare_dot4(inputs[13],inputs[14],inputs[15]);
A result4 = inputs[0].prepare_dot4(inputs[1],inputs[2],inputs[3]) + inputs[4].prepare_dot4(inputs[5],inputs[6],inputs[7]) + inputs[8].prepare_dot4(inputs[9],inputs[10],inputs[11]) + inputs[12].prepare_dot4(inputs[13],inputs[14],inputs[15]);
A result5 = inputs[0].prepare_dot4(inputs[1],inputs[2],inputs[3]) + inputs[4].prepare_dot4(inputs[5],inputs[6],inputs[7]) + inputs[8].prepare_dot4(inputs[9],inputs[10],inputs[11]) + inputs[12].prepare_dot4(inputs[13],inputs[14],inputs[15]);
A result6 = inputs[0].prepare_dot4(inputs[1],inputs[2],inputs[3]) + inputs[4].prepare_dot4(inputs[5],inputs[6],inputs[7]) + inputs[8].prepare_dot4(inputs[9],inputs[10],inputs[11]) + inputs[12].prepare_dot4(inputs[13],inputs[14],inputs[15]);
A result7 = inputs[0].prepare_dot4(inputs[1],inputs[2],inputs[3]) + inputs[4].prepare_dot4(inputs[5],inputs[6],inputs[7]) + inputs[8].prepare_dot4(inputs[9],inputs[10],inputs[11]) + inputs[12].prepare_dot4(inputs[13],inputs[14],inputs[15]);
A result8 = inputs[0].prepare_dot4(inputs[1],inputs[2],inputs[3]) + inputs[4].prepare_dot4(inputs[5],inputs[6],inputs[7]) + inputs[8].prepare_dot4(inputs[9],inputs[10],inputs[11]) + inputs[12].prepare_dot4(inputs[13],inputs[14],inputs[15]);
result1.mask_and_send_dot_without_trunc();
result2.mask_and_send_dot_without_trunc();
result3.mask_and_send_dot_without_trunc();
result4.mask_and_send_dot_without_trunc();
result5.mask_and_send_dot_without_trunc();
result6.mask_and_send_dot_without_trunc();
result7.mask_and_send_dot_without_trunc();
result8.mask_and_send_dot_without_trunc();
Share::communicate();
result1.complete_mult();
result2.complete_mult();
result3.complete_mult();
result4.complete_mult();
result5.complete_mult();
result6.complete_mult();
result7.complete_mult();
result8.complete_mult();
result1 = result1.prepare_dot4(result2,result3,result4) + result5.prepare_dot4(result6,result7,result8);
result1.mask_and_send_dot();
Share::communicate();
result1.complete_mult();
result1.prepare_reveal_to_all();
Share::communicate();
DATATYPE* result_arr = new DATATYPE;
result_arr[0] = result1.complete_reveal_to_all();
#elif FUNCTION_IDENTIFIER == 34 // dot234 mixed
A result1 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot3(inputs[3],inputs[4]) + inputs[5].prepare_dot4(inputs[6],inputs[7],inputs[8]);
A result2 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot3(inputs[3],inputs[4]) + inputs[5].prepare_dot4(inputs[6],inputs[7],inputs[8]);
A result3 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot3(inputs[3],inputs[4]) + inputs[5].prepare_dot4(inputs[6],inputs[7],inputs[8]);
A result4 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot3(inputs[3],inputs[4]) + inputs[5].prepare_dot4(inputs[6],inputs[7],inputs[8]);
A result5 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot3(inputs[3],inputs[4]) + inputs[5].prepare_dot4(inputs[6],inputs[7],inputs[8]);
A result6 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot3(inputs[3],inputs[4]) + inputs[5].prepare_dot4(inputs[6],inputs[7],inputs[8]);
A result7 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot3(inputs[3],inputs[4]) + inputs[5].prepare_dot4(inputs[6],inputs[7],inputs[8]);
A result8 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot3(inputs[3],inputs[4]) + inputs[5].prepare_dot4(inputs[6],inputs[7],inputs[8]);
A result9 = inputs[0].prepare_dot(inputs[1]) + inputs[2].prepare_dot3(inputs[3],inputs[4]) + inputs[5].prepare_dot4(inputs[6],inputs[7],inputs[8]);
result1.mask_and_send_dot_without_trunc();
result2.mask_and_send_dot_without_trunc();
result3.mask_and_send_dot_without_trunc();
result4.mask_and_send_dot_without_trunc();
result5.mask_and_send_dot_without_trunc();
result6.mask_and_send_dot_without_trunc();
result7.mask_and_send_dot_without_trunc();
result8.mask_and_send_dot_without_trunc();
result9.mask_and_send_dot_without_trunc();
Share::communicate();
result1.complete_mult_without_trunc();
result2.complete_mult_without_trunc();
result3.complete_mult_without_trunc();
result4.complete_mult_without_trunc();
result5.complete_mult_without_trunc();
result6.complete_mult_without_trunc();
result7.complete_mult_without_trunc();
result8.complete_mult_without_trunc();
result9.complete_mult_without_trunc();
result1 = result1.prepare_dot(result2) + result3.prepare_dot3(result4,result5) + result6.prepare_dot4(result7,result8,result9);
result1.mask_and_send_dot_without_trunc();
Share::communicate();
result1.complete_mult_without_trunc();
result1.prepare_reveal_to_all();
Share::communicate();
DATATYPE* result_arr = new DATATYPE;
result_arr[0] = result1.complete_reveal_to_all();
#endif
if(current_phase == PHASE_LIVE)
{
    std::cout << "P" << PARTY << " result: " << std::to_string(result_arr[0]) << std::endl;
}


}


#endif

#if FUNCTION_IDENTIFIER == 35

template<int k,typename Share, typename Datatype>
void RELU_range_in_place(sint_t<Additive_Share<Datatype, Share>>* val, int len)
{
    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<k, S>;
    using sint = sint_t<A>;
    
    Bitset *s1 = new Bitset[NUM_INPUTS];
    Bitset *s2 = new Bitset[NUM_INPUTS];
    for(int i = 0; i < len; i++)
    {
        s1[i] = Bitset::prepare_A2B_S1( (S*) val[i].get_share_pointer());
        s2[i] = Bitset::prepare_A2B_S2( (S*) val[i].get_share_pointer());
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        s1[i].complete_A2B_S1();
        s2[i].complete_A2B_S2();
    }
    /* Bitset* y = new Bitset[NUM_INPUTS]; */
    S *y = new S[len];
    /* BooleanAdder<S> *adder = new BooleanAdder<S>[NUM_INPUTS]; */
    std::vector<BooleanAdder_MSB<k,S>> adders;
    /* std::vector<PPA_MSB_4Way<k,S>> adders; */
    adders.reserve(len);
    for(int i = 0; i < len; i++)
    {
        /* adder[i].set_values(s1[i], s2[i], y[i]); */
        adders.emplace_back(s1[i], s2[i], y[i]);
    }
    while(!adders[0].is_done())
    {
        for(int i = 0; i < len; i++)
        {
            adders[i].step();
        }
        Share::communicate();
    }
    delete[] s1;
    delete[] s2;
    adders.clear();
    adders.shrink_to_fit();
    
    for(int i = 0; i < len; i++)
    {
        y[i] = ~ y[i];
    }
    sint* t1 = new sint[len];
    sint* t2 = new sint[len];
    for(int i = 0; i < len; i++)
    {
        y[i].prepare_bit_injection_S1(t1[i].get_share_pointer());
        y[i].prepare_bit_injection_S2(t2[i].get_share_pointer());
    }
    delete[] y;
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        t1[i].complete_bit_injection_S1();
        t2[i].complete_bit_injection_S2();
    }
    sint* result = new sint[len];
    for(int i = 0; i < len; i++)
    {
        result[i].prepare_XOR(t1[i],t2[i]);
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        result[i].complete_XOR(t1[i],t2[i]);
    }
    delete[] t1;
    delete[] t2;

    for(int i = 0; i < len; i++)
    {
        val[i] = result[i] * val[i];
    }
    delete[] result;
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        val[i].complete_mult();
    }
}
    
    template<typename Share>
void RELU_range_test(DATATYPE* res)
{
using A = Additive_Share<DATATYPE, Share>;
using sint = sint_t<A>;
auto a = new sint[NUM_INPUTS];
for(int i = 0; i < NUM_INPUTS; i++)
        a[i].template prepare_receive_from<P_0>();
Share::communicate();
for(int i = 0; i < NUM_INPUTS; i++)
        a[i].template complete_receive_from<P_0>();
Share::communicate();
RELU_range_in_place<REDUCED_BITLENGTH,Share>(a,NUM_INPUTS);
auto result_arr = NEW(UINT_TYPE[NUM_INPUTS][DATTYPE]);
for(int i = 0; i < NUM_INPUTS; i++)
    a[i].prepare_reveal_to_all();
Share::communicate();
for(int i = 0; i < NUM_INPUTS; i++)
    a[i].complete_reveal_to_all(result_arr[i]);
for(int i = 0; i < NUM_INPUTS; i++)
    if(current_phase == PHASE_LIVE)
    std::cout << "P" << PARTY << ": Result " << i << ": "<< result_arr[i][0] << std::endl;
}
#endif

#if FUNCTION_IDENTIFIER == 39
template<typename Share>
void mat_mul_test(DATATYPE *res)
{
    using A = Additive_Share<DATATYPE, Share>;
    const int m = 3;
    const int k = 2;
    const int n = 1;
    /* auto X = new A[m*k]{3,5,7,11,2,4}; //initialize shares with public values */
    /* auto W = new A[k*n]{2,4}; */
    /* auto Y = new A[m*n]{0,0,0}; */
    auto X = new A[m*k]{A(3),A(5),A(7),A(11),A(2),A(4)}; //initialize shares with public values
    auto W = new A[k*n]{A(2),A(4)}; 
    auto Y = new A[m*n]{A(0),A(0),A(0)};
    Share::communicate();

#if USE_CUDA_GEMM == 1
    /* auto XS = new Share[m*k]; */
    /* auto WS = new Share[k*n]; */
    /* auto YS = new Share[m*n]; */
    /* for(int i = 0; i < m*k; i++) */
    /*     XS[i] = X[i].get_share(); */
    /* for(int i = 0; i < k*n; i++) */
    /*     WS[i] = W[i].get_share(); */
    /* for(int i = 0; i < m*n; i++) */
    /*     YS[i] = Y[i].get_share(); */
    /* A::GEMM(XS,WS,YS,m,n,k,OP_ADD,OP_SUB,OP_MULT); // RowX, ColW, ColX, X, W, Y */
    A::GEMM(X,W,Y,m,n,k); // RowX, ColW, ColX, X, W, Y
    /* for(int i = 0; i < m*n; i++) */
    /*     Y[i].set_share(YS[i]); */
    /* delete[] XS; */
    /* delete[] WS; */
    /* delete[] YS; */

#else
    //Naive Mat Mul for correctness test
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            for(int q = 0; q < k; q++)
            {
                Y[i*n+j] = Y[i*n+j] + X[i*k+q] * W[q*n+j];
            }
        }
    }
#endif
    for(int i = 0; i < m*n; i++)
    {
        Y[i].mask_and_send_dot_without_trunc(); //no truncation because values are not fixed point
    }
    Share::communicate();
    for(int i = 0; i < m*n; i++)
    {
        Y[i].complete_mult_without_trunc();
    }

    for(int i = 0; i < m*n; i++)
    {
        Y[i].prepare_reveal_to_all();
    }
    Share::communicate();
    UINT_TYPE result_arr[m*n][DATTYPE/BITLENGTH];
    for(int i = 0; i < m*n; i++)
    {
        Y[i].complete_reveal_to_all(result_arr[i]);
    }
    for(int i = 0; i < m*n; i++)
    {
        if(current_phase == PHASE_LIVE)
            std::cout << "P" << PARTY << ": Result " << i << ": "<< result_arr[i][0] << std::endl;
    }
}
#endif
