#pragma once
#include "../../protocols/Protocols.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <bitset>
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

#include <cmath>

/* #include "boolean_adder.hpp" */
/* #include "ppa.hpp" */
#if FUNCTION_IDENTIFIER == 16 || FUNCTION_IDENTIFIER == 17
#define FUNCTION RELU_bench
#elif FUNCTION_IDENTIFIER == 19
#define FUNCTION AND_bench
#elif FUNCTION_IDENTIFIER == 18
#define FUNCTION fixed_test
#elif FUNCTION_IDENTIFIER == 13
#define FUNCTION dot_prod_bench
#elif FUNCTION_IDENTIFIER == 14
#define FUNCTION dot_prod_bench
/* #define FUNCTION matmul_bench */
#elif FUNCTION_IDENTIFIER == 15
#define FUNCTION conv2D
#elif FUNCTION_IDENTIFIER == 20 || FUNCTION_IDENTIFIER == 23 || FUNCTION_IDENTIFIER == 25
#define FUNCTION forward_pass
#define USE_EIGEN 1
#elif FUNCTION_IDENTIFIER == 21 || FUNCTION_IDENTIFIER == 24 || FUNCTION_IDENTIFIER == 26
#define FUNCTION backward_pass
#define USE_EIGEN 1
#elif FUNCTION_IDENTIFIER == 22
#define FUNCTION FC_bench
#define USE_EIGEN 1
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
#elif FUNCTION_IDENTIFIER == 39
#include "comp_trunc.hpp"
#define FUNCTION test_comp_trunc
#elif FUNCTION_IDENTIFIER >= 37 && FUNCTION_IDENTIFIER <= 57
#define USE_EIGEN 1
#define FUNCTION conv2D_bench
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
            /* if(current_phase == 1) */
            /* { */
            /* #if FRACTIONAL > 0 */
            /* std::cout << fixedToFloat<float, UINT_TYPE, FRACTIONAL>(result_arr[i][j][0]) << std::endl; */
            /* #else */
            /* std::cout << result_arr[2+i+j][0] << std::endl; */
            /* #endif */
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
    using M = Matrix_Share<DATATYPE, Share>;
    auto a = new M[NUM_INPUTS];
    auto b = new M[NUM_INPUTS][NUM_INPUTS];
    auto c = new M[NUM_INPUTS];
    Share::communicate(); // dummy round
    for(int i = 0; i < NUM_INPUTS; i++)
    {
#if FUNCTION_IDENTIFIER == 14
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            c[i] += a[i] * b[i][j];
        }
#endif
        c[i].mask_and_send_dot();
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
            c[i].complete_mult();
    
}

    Share::communicate();
    c[NUM_INPUTS-1].prepare_reveal_to_all();
    Share::communicate();
    *res = c[NUM_INPUTS-1].complete_reveal_to_all();

delete[] a;
delete[] b;
delete[] c;

}


    /* template<typename Share> */
/* void matmul(DATATYPE* res) */
/* { */
    /* using S = XOR_Share<DATATYPE, Share>; */
    /* using A = Additive_Share<DATATYPE, Share>; */
    /* using M = Matrix_Share<DATATYPE, Share>; */
    /* using Bitset = sbitset_t<BITLENGTH,S>; */
    /* using sint = sint_t<A>; */
    /* using mint = sint_t<M>; */

    /* Eigen::Matrix<mint, 2, 2> mat1, mat2, result; */

    /* //Receive shares */

    /* /1* mat1 << mint().template prepare_receive_from<P_0>(), mint().template prepare_receive_from<P_0>(), *1/ */
    /* /1*         mint().template prepare_receive_from<P_0>(), mint().template prepare_receive_from<P_0>(); *1/ */
    
    /* /1* mat2 << mint().template prepare_receive_from<P_1>(), mint().template prepare_receive_from<P_1>(), *1/ */
    /* /1*         mint().template prepare_receive_from<P_1>(), mint().template prepare_receive_from<P_1>(); *1/ */
    
    /* for(int i = 0; i < 2; i++) */
    /* { */
    /*     for(int j = 0; j < 2; j++) */ 
    /*     { */
    /*         mat1(i, j). template prepare_receive_from<P_0>(); */
    /*         mat2(i, j). template prepare_receive_from<P_1>(); */
    /*     } */
    /* } */


    /* Share::communicate(); */

    /* //complete_receive */

    /* for(int i = 0; i < 2; i++) */
    /* { */
    /*     for(int j = 0; j < 2; j++) */ 
    /*     { */
    /*         mat1(i, j). template complete_receive_from<P_0>(); */
    /*         mat2(i, j). template complete_receive_from<P_1>(); */
    /*     } */
    /* } */

    /* // 5. Multiply matrices. */
    /* result = mat1 * mat2; */

    /* // send dot product */

    /* for(int i = 0; i < 2; i++) */
    /* { */
    /*     for(int j = 0; j < 2; j++) */ 
    /*     { */
    /*         result(i, j).mask_and_send_dot(); */
    /*     } */
    /* } */

    /* Share::communicate(); */

    /* /1* //complete_receive and reveal *1/ */
    
    /* for(int i = 0; i < 2; i++) */
    /* { */
    /*     for(int j = 0; j < 2; j++) */ 
    /*     { */
    /*         result(i, j).complete_mult(); */
    /*         result(i,j).prepare_reveal_to_all(); */
    /*         /1* mat2(i,j).prepare_reveal_to_all(); *1/ */
    /*     } */
    /* } */

    /* Share::communicate(); */

    /* UINT_TYPE result_arr[4][DATTYPE]; */
    /* for(int i = 0; i < 2; i++) */
    /* { */
    /*     for(int j = 0; j < 2; j++) */ 
    /*     { */
    /*         result(i, j).complete_reveal_to_all(result_arr[2*i+j]); */
    /*         /1* mat2(i,j).complete_reveal_to_all(result_arr[2*i+j]); *1/ */
    /*     } */
    /* } */


    
    /* if(current_phase == 1) */
    /* { */
    /*     std::cout << "P" << PARTY << ": Result: "; */
    /*     for (int i = 0; i < 4; i++) */
    /*     { */
    /* for(int j = 0; j < DATTYPE; j++) */
    /* { */
    /*     std::cout << result_arr[i][j] << " "; */
    /* std::cout << std::endl; */
    /* } */
    /* std::cout << std::endl; */
/* } */
    

/* } */
/* } */
/* template<typename Share> */
/* void conv2D(DATATYPE* res) */
/* { */
/*     using M = Matrix_Share<DATATYPE, Share>; */
/*     using Matrix = Eigen::Matrix<sint_t<M>, Eigen::Dynamic, Eigen::Dynamic>; */

/*     const int depth = 64; */
/*     const int kernel_size = 3; */

/*     // Input (with padding) */
/*     std::vector<Matrix> input(depth, Matrix(34, 34)); */

/*     // Kernel */
/*     std::vector<Matrix> kernel(depth, Matrix(3, 3)); */

/*     // Output */
/*     std::vector<Matrix> output(depth, Matrix(32, 32)); */

/*     for (int z = 0; z < depth; ++z) { */
/*         for (int i = 0; i <= 32 - kernel_size; ++i) { */
/*             for (int j = 0; j <= 32 - kernel_size; ++j) { */
/*                 for (int d = 0; d < depth; ++d) { */
/*                     // Extract the patch from the input using block operation */
/*                     Matrix patch = input[d].block(i, j, kernel_size, kernel_size); */
/*                     output[z](i, j) += (patch.array() * kernel[d].array()).sum(); */
/*                 } */
/*             } */
/*         } */
/*     } */

/*      for (int z = 0; z < depth; ++z) { */
/*         for (int i = 0; i < 32; ++i) { */
/*             for (int j = 0; j < 32; ++j) { */
/*                 output[z](i, j).mask_and_send_dot(); */
/*             } */
/*         } */
/*     } */

/*      Share::communicate(); */

/*           for (int z = 0; z < depth; ++z) { */
/*         for (int i = 0; i < 32; ++i) { */
/*             for (int j = 0; j < 32; ++j) { */
/*                 output[z](i, j).complete_mult(); */
/*             } */
/*         } */
/*     } */

/* } */

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
        result[i] = result[i] * val[i];
    }
    delete[] val;
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        result[i].complete_mult();
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




#if FUNCTION_IDENTIFIER > 19 && USE_EIGEN == 1
#include <eigen3/Eigen/Core>
template<typename T>
using MatX = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template<typename T>
using VecX = Eigen::Matrix<T, Eigen::Dynamic, 1>;


#include <vector>

using namespace std;
using namespace Eigen;

	int calc_outsize(int in_size, int kernel_size, int stride, int pad)
	{
		return (int)std::floor((in_size + 2 * pad - kernel_size) / stride) + 1;
	}

    template <typename T>
    T im2col_get_pixel(const T* im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return T(0); // public value, no rand needed
    return im[col + width * (row + height * channel)];
}

template <typename T>
void col2im_add_pixel(T* im, int height, int width, int channels,
                    int row, int col, int channel, int pad, T val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return;
    im[col + width * (row + height * channel)] += val;
}

// This one might be too, can't remember.

template <typename T>
void col2im(const T* data_col, int channels, int height, int width,
            int ksize, int stride, int pad, T* data_im)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                T val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad, val);
            }
        }
    }
}


// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE
template <typename T>
void im2col(const T* data_im, int channels, int height, int width,
            int ksize, int stride, int pad, T* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}




enum class LayerType
	{
		LINEAR,
		CONV2D,
		MAXPOOL2D,
		AVGPOOL2D,
		ACTIVATION,
		BATCHNORM1D,
		BATCHNORM2D,
		FLATTEN
	};
template<typename T>
class Layer
{
public:
    LayerType type;
    bool is_first;
    bool is_last;
    MatX<T> output;
    MatX<T> delta;
public:
    Layer(LayerType type) : type(type), is_first(false), is_last(false) {}
    virtual void set_layer(const vector<int>& input_shape) = 0;
    virtual void forward(const MatX<T>& prev_out, bool is_training = true) = 0;
    virtual void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) = 0;
    /* virtual void update_weight(T lr, T decay) { return; } */
    /* virtual void zero_grad() { return; } */
    /* virtual vector<int> output_shape() = 0; */
};


template<typename T> 
class Conv2d : public Layer<T>
	{
	private:
		int batch;
		int ic;
		int oc;
		int ih;
		int iw;
		int ihw;
		int oh;
		int ow;
		int ohw;
		int kh;
		int kw;
		int pad;
		string option;
		MatX<T> dkernel;
		VecX<T> dbias;
		MatX<T> im_col;
	public:
		MatX<T> kernel;
		VecX<T> bias;
		Conv2d(int in_channels, int out_channels, int kernel_size, int padding,
			string option);
		void set_layer(const vector<int>& input_shape) override;
		void forward(const MatX<T>& prev_out, bool is_training) override;
		void forward1(const MatX<T>& prev_out, bool is_training);
		void forward2(const MatX<T>& prev_out, bool is_training);
		void forward3(const MatX<T>& prev_out, bool is_training);
		void forward4(const MatX<T>& prev_out, bool is_training);
		void forward5(const MatX<T>& prev_out, bool is_training);
        void forward6(const MatX<T>& prev_out, bool is_training);
        void forward7(const MatX<T>& prev_out, bool is_training);
        void forward8(const MatX<T>& prev_out, bool is_training);
        void forward9(const MatX<T>& prev_out, bool is_training);
        void forward10(const MatX<T>& prev_out, bool is_training);
        void forward11(const MatX<T>& prev_out, bool is_training);
        void forward12(const MatX<T>& prev_out, bool is_training);
        void forward13(const MatX<T>& prev_out, bool is_training);
        void forward14(const MatX<T>& prev_out, bool is_training);
        void forward15(const MatX<T>& prev_out, bool is_training);
        void forward16(const MatX<T>& prev_out, bool is_training);
        void forward17(const MatX<T>& prev_out, bool is_training);
        void forward18(const MatX<T>& prev_out, bool is_training);
        void forward18_old(const MatX<T>& prev_out, bool is_training);
        void forward19(const MatX<T>& prev_out, bool is_training);
        void forward20(const MatX<T>& prev_out, bool is_training);
        void forward21(const MatX<T>& prev_out, bool is_training);
        void forward22(const MatX<T>& prev_out, bool is_training);
        void forward23(const MatX<T>& prev_out, bool is_training);
        void forward24(const MatX<T>& prev_out, bool is_training);
        void forward25(const MatX<T>& prev_out, bool is_training);
        void forward26(const MatX<T>& prev_out, bool is_training);
        void forward27(const MatX<T>& prev_out, bool is_training);
        void forward28(const MatX<T>& prev_out, bool is_training);
        void forward29(const MatX<T>& prev_out, bool is_training);
        void forward30(const MatX<T>& prev_out, bool is_training);
        void forward31(const MatX<T>& prev_out, bool is_training);
        void forward32(const MatX<T>& prev_out, bool is_training);
        void forward33(const MatX<T>& prev_out, bool is_training);
        void forward34(const MatX<T>& prev_out, bool is_training);
		void backward(const MatX<T>& prev_out, MatX<T>& prev_delta) override;
		/* void update_weight(T lr, T decay) override; */
		/* void zero_grad() override; */
		/* vector<int> output_shape() override; */
	};

    template<typename T>
	Conv2d<T>::Conv2d(
		int in_channels,
		int out_channels,
		int kernel_size,
		int padding,
		string option
	) :
		Layer<T>(LayerType::CONV2D),
		batch(0),
		ic(in_channels),
		oc(out_channels),
		ih(0),
		iw(0),
		ihw(0),
		oh(0),
		ow(0),
		ohw(0),
		kh(kernel_size),
		kw(kernel_size),
		pad(padding),
		option(option) {}

    template<typename T>
	void Conv2d<T>::set_layer(const vector<int>& input_shape)
	{
		batch = input_shape[0];
		ic = input_shape[1];
		ih = input_shape[2];
		iw = input_shape[3];
		ihw = ih * iw;
		oh = calc_outsize(ih, kh, 1, pad);
		ow = calc_outsize(iw, kw, 1, pad);
		ohw = oh * ow;

		this->output.resize(batch * oc, ohw);
		this->delta.resize(batch * oc, ohw);
		kernel.resize(oc, ic * kh * kw);
		dkernel.resize(oc, ic * kh * kw);
		bias.resize(oc);
		dbias.resize(oc);
		im_col.resize(ic * kh * kw, ohw);

		int fan_in = kh * kw * ic;
		int fan_out = kh * kw * oc;
		/* init_weight(kernel, fan_in, fan_out, option); */
		/* bias.setZero(); */
	}


int TILE_SIZE = 64;
    template<typename T>
	void Conv2d<T>::forward(const MatX<T>& prev_out, bool is_training)
	{
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			this->output.block(oc * n, 0, oc, ohw).noalias() = kernel * im_col;
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
		}
	}
   
//Eigen
    template<typename T>
	void Conv2d<T>::forward1(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			this->output.block(oc*n,0, oc, ohw).noalias() = kernel * im_col;
        }
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).mask_and_send_dot();
            }
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
		
	}
    
    template<typename T>
	void Conv2d<T>::forward2(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            for(int i = 0; i < oc; ++i) {
        for(int k = 0; k < kernel.cols(); ++k) {
            T temp = kernel(i, k);
                for(int j = 0; j < ohw; ++j) {
                    this->output(oc * n + i, j) += temp * im_col(k, j);  // Use custom * and + operators
                    }
                }
                for(int j = 0; j < ohw; ++j) 
                    this->output(oc * n + i, j).mask_and_send_dot();
        }
        }
        /* for (int j = 0; j < this->output.size(); j++) { */
        /*     this->output(j).mask_and_send_dot(); */
        /* } */

            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
	}
    
            template<typename T>
	void Conv2d<T>::forward3(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
        for(int k = 0; k < kernel.cols(); ++k) {
            for(int i = 0; i < oc; ++i) {
                for(int j = 0; j < ohw; ++j) {
                    this->output(oc *n + i, j) += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators
                    }
                }
        }
        }
        for (int j = 0; j < this->output.size(); j++) {
            this->output(j).mask_and_send_dot();
        }

            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
	}
      // naive      
            template<typename T>
	void Conv2d<T>::forward4(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            for(int i = 0; i < oc; ++i) {
                T sum = T(0);
                for(int j = 0; j < ohw; ++j) {
                        for(int k = 0; k < kernel.cols(); ++k) {
                    sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators
                    }
                sum.mask_and_send_dot();
                this->output(oc * n + i, j) = sum;
                }
        }
    }

            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
	}
            
            template<typename T>
	void Conv2d<T>::forward5(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
    auto A = kernel;
    auto B = im_col;
    auto C = this->output;
    const int Mtile = 64; // tile size in M dimension
    const int Ntile = 64; // tile size in N dimension
    const int M = oc;
    const int N = ohw; 
    const int K = kernel.cols(); 

    for (int m = 0; m < M; m += Mtile)                // iterate over M dimension
    {
    for (int q = 0; q < N; q += Ntile)            // iterate over N dimension
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < Mtile; ++i)       // compute one tile 
                for (int j = 0; j < Ntile; ++j) {
                    int row = m + i;
                    int col = q + j;
                    C(n*oc + row,col) += A(row,k) * B(k,col);
                    /* C[row][col] += A[row][k] * B[k][col]; */
                }
    }
        }
    
        for (int i = 0; i < this->output.size(); i++) {
            this->output(i).mask_and_send_dot();
    }
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
            template<typename T>
	void Conv2d<T>::forward6(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
    auto A = kernel;
    auto B = im_col;
    auto C = this->output;
    const int Mtile = 64; // tile size in M dimension
    const int Ntile = 64; // tile size in N dimension
    const int M = oc;
    const int N = ohw; 
    const int K = kernel.cols(); 

    for (int m = 0; m < M; m += Mtile)                // iterate over M dimension
    {
    for (int q = 0; q < N; q += Ntile)            // iterate over N dimension
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < Mtile; ++i)       // compute one tile 
                    {
                    int row = m + i;
                    T temp = A(row,k);
                for (int j = 0; j < Ntile; ++j) {
                    int col = q + j;
                    C(n*oc + row,col) += temp * B(k,col);
                    /* C[row][col] += A[row][k] * B[k][col]; */
                }
                }
    }
        }
    
        for (int i = 0; i < this->output.size(); i++) {
            this->output(i).mask_and_send_dot();
    }
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

            template<typename T>
	void Conv2d<T>::forward7(const MatX<T>& prev_out, bool is_training)
	{
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
    auto A = kernel;
    auto B = im_col;
    auto C = this->output;
    const int Mtile = 64; // tile size in M dimension
    const int Ntile = 64; // tile size in N dimension
    const int M = oc;
    const int N = ohw; 
    const int K = kernel.cols(); 

    for (int m = 0; m < M; m += Mtile)                // iterate over M dimension
    {
    for (int q = 0; q < N; q += Ntile)            // iterate over N dimension
        for (int k = 0; k < K; ++k)
                for (int j = 0; j < Ntile; ++j) {
                    int col = q + j;
                    T temp = B(k,col);
            for (int i = 0; i < Mtile; ++i)       // compute one tile 
                    {
                    int row = m + i;
                    C(n*oc + row,col) += A(row,k) * temp;
                    /* C[row][col] += A[row][k] * B[k][col]; */
                }
                }
    }
        }
    
        for (int i = 0; i < this->output.size(); i++) {
            this->output(i).mask_and_send_dot();
    }
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
            
    template<typename T>
	void Conv2d<T>::forward8(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */
    auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    /* const int m = kernel.rows(); */
    /* const int f = kernel.cols(); */
    /* const int p = im_col.cols(); */
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;

  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    for (int jj = j; jj < j_max; ++jj) {
                        for (int kk = k; kk < k_max; ++kk) {
                            /* C(oc * n + ii,jj) += A(ii,kk) * B(kk,jj); */
                            /* C[ii][jj] += A[ii][kk] * B[kk][jj]; */
                       C[ii * p + jj] += A[ii * f + kk] * B[kk * p + jj]; 
                        }
                    }
                }
            }
                for (int ii = i; ii < i_max; ++ii) 
                    for (int jj = j; jj < j_max; ++jj) 
                        C[ii * p + jj].mask_and_send_dot();
    }
}
} 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    
    template<typename T>
	void Conv2d<T>::forward9(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
                for (int ii = i; ii < i_max; ++ii) {
                    for (int jj = j; jj < j_max; ++jj) {
                        for (int k = 0; k < f; k++) {
                /* int k_max = std::min(k + TILE_SIZE, f); */
                /*         for (int kk = k; kk < k_max; ++kk) { */
                            /* for (int k = 0; k < f; k ++) { */
                       C[ii * p + jj] += A[ii * f + k] * B[k * p + jj]; 
                        }
                    C[ii * p + jj].mask_and_send_dot();
                    }
                }
            /* for (int ii = i; ii < i_max; ++ii) { */
            /*     for (int jj = j; jj < j_max; ++jj) { */
            /*         C[ii * p + jj].mask_and_send_dot(); */
            /*     } */
            /* } */
            }
        }
    




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward10(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */
    auto A = kernel.data();
    /* auto B = im_col.transpose().data(); */
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    T temp = A[ii*f+kk]; 
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii * p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                for (int jj = j; jj < j_max; ++jj) {
                    C[ii * p + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    
    template<typename T>
	void Conv2d<T>::forward11(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                        for (int kk = k; kk < k_max; ++kk) {
                    const int row2 = kk*p;
                for (int ii = i; ii < i_max; ++ii) {
                    const int row = ii*p;
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[row + jj] += temp * B[row2 + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    
    template<typename T>
	void Conv2d<T>::forward12(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
          for (int i = 0; i < n; i ++) {
                    for (int jj = j; jj < j_max; ++jj) {
                        T temp = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                    temp += A[i*f+kk] * B[kk*p + jj]; 
                        }
                        temp.mask_and_send_dot();
                       C[i * p + jj] += temp; 
    }
  }
            }
        }
        } 




        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward13(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
       
for (int i = 0; i < m; i += TILE_SIZE) {
    int i_max = std::min(i + TILE_SIZE, m);
    for (int k = 0; k < f; k += TILE_SIZE) {
        int k_max = std::min(k + TILE_SIZE, f);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);

            for (int ii = i; ii < i_max; ++ii) {
                for (int kk = k; kk < k_max; ++kk) {
                    auto temp = A[ii * f + kk];
                    for (int jj = j; jj < j_max; ++jj) {
                        C[ii * p + jj] += temp * B[kk * p + jj];
                    }
                }
            }
        }
    }
}

// Post-processing step
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
        C[i * p + j].mask_and_send_dot();
    }
}

}
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }


    template<typename T>
	void Conv2d<T>::forward14(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
        

for (int i0 = 0; i0 < m; i0 += TILE_SIZE) {
    int i_max = std::min(i0 + TILE_SIZE, m);
    for (int k0 = 0; k0 < f; k0 += TILE_SIZE) {
        int k_max = std::min(k0 + TILE_SIZE, f);
        for (int j0 = 0; j0 < p; j0 += TILE_SIZE) {
            int j_max = std::min(j0 + TILE_SIZE, p);

            for (int i = i0; i < i_max; ++i) {
                for (int j = j0; j < j_max; ++j) {
                    T temp_sum = T(0);
                    for (int k = k0; k < k_max; ++k) {
                        temp_sum += A[i * f + k] * B[k * p + j];
                    }
                    C[i * p + j] += temp_sum;
                }
            }
        }
    }
}

// Post-processing step
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
        C[i * p + j].mask_and_send_dot();
    }
}





        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    
    template<typename T>
	void Conv2d<T>::forward15(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    const int row = ii*p;
                        for (int kk = k; kk < k_max; ++kk) {
                    const int row2 = kk*p;
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[row + jj] += temp * B[row2 + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    template<typename T>
	void Conv2d<T>::forward16(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                        for (int kk = k; kk < k_max; ++kk) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

        

    template<typename T>
	void Conv2d<T>::forward17(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    // all opt
    template<typename T>
	void Conv2d<T>::forward18(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */
            auto A = kernel.data();
    /* auto B = im_col.transpose().data(); */
        auto BM = im_col.transpose();
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
      /* _mm_prefetch(A + i * f, _MM_HINT_T0); */
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            C[i * p + j] = T(0);
            /* _mm_prefetch(B + j * f, _MM_HINT_T0); */
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    const int iif = ii*f;
                    const int iip = ii*p;

                    /* const int row2 = ii*f+kk; */
                    for (int jj = j; jj < j_max; ++jj) {
                    auto temp = T(0);
                        const int jjf = jj*f;
                        for (int kk = k; kk < k_max; ++kk) {
                            /* _mm_prefetch(C + ii * p + jj, _MM_HINT_T0); */
                       temp += A[iif+kk] * B[jjf + kk]; 
                        }
                        C[iip + jj] += temp;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }

        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            /* T::communicate(); */
            /* for (int i = 0; i < this->output.size(); i++) { */
            /*     this->output(i).complete_mult(); */
            /* } */
        T::communicate();
            /* auto b = bias.data(); */
            const int m = oc;
            const int p = ohw;
		for (int n = 0; n < batch; n++) {
            auto C = this->output.data() + (oc * ohw) * n;
            for( int i = 0; i < m; i += TILE_SIZE) {
                const int i_max = std::min(i + TILE_SIZE, m);
                for (int j = 0; j < p; j += TILE_SIZE) {
                    const int j_max = std::min(j + TILE_SIZE, p);
                    for (int ii = i; ii < i_max; ++ii) {
                        const int row = ii*p;
                        for (int jj = j; jj < j_max; ++jj) {
                            C[row + jj].complete_mult();
                            /* C[row + jj] += b[row+jj]; */
                        }
                    }
                }
            }
    }
		for (int n = 0; n < batch; n++)
            this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
    }
    template<typename T>
	void Conv2d<T>::forward18_old(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */
            auto A = kernel.data();
    auto BM = im_col.transpose();
        /* MatX<T> BM = im_col.transpose(); */
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
      /* _mm_prefetch(A + i * f, _MM_HINT_T0); */
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            /* _mm_prefetch(B + j * f, _MM_HINT_T0); */
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    for (int jj = j; jj < j_max; ++jj) {
                    auto temp = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                            /* _mm_prefetch(C + ii * p + jj, _MM_HINT_T0); */
                       temp += A[ii*f+kk] * B[jj*f + kk]; 
                        }
                        C[ii*p + jj] += temp;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }

        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward19(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */
            auto A = kernel.data();
    auto B = im_col.transpose().data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = T(0);
                    for (int jj = j; jj < j_max; ++jj) {
                        for (int k = 0; k < f; k ++) {
                       temp += A[ii*f+k] * B[jj*f + k]; 
                        }
                        temp.mask_and_send_dot();
                        C[ii*p + jj] = temp;
                    }
                }
            }
            }

        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    //transpose
    template<typename T>
	void Conv2d<T>::forward20(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    /* auto B = im_col.transpose().data(); */
    auto BM = im_col.transpose();
        /* MatX<T> BM = im_col.transpose(); */
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i ++) {
        for (int j = 0; j < p; j ++) {
                    auto temp = T(0);
                        for (int k = 0; k < f; k ++) {
                       temp += A[i*f+k] * B[j*f + k]; 
                        }
                        temp.mask_and_send_dot();
                        C[i*p + j] = temp;
                    }
                }
            }
  /* for (int i = 0; i < m; i += TILE_SIZE) { */
  /*       int i_max = std::min(i + TILE_SIZE, m); */
  /*       for (int j = 0; j < p; j += TILE_SIZE) { */
  /*           int j_max = std::min(j + TILE_SIZE, p); */
  /*           for (int k = 0; k < f; k += TILE_SIZE) { */
  /*               int k_max = std::min(k + TILE_SIZE, f); */
  /*               for (int ii = i; ii < i_max; ++ii) { */
  /*                       for (int kk = k; kk < k_max; ++kk) { */
  /*                   /1* const int row2 = ii*f+kk; *1/ */
  /*                   auto temp = A[ii*f+kk]; */
  /*                   for (int jj = j; jj < j_max; ++jj) { */
  /*                      C[ii*p + jj] += temp * B[jj*f + kk]; */ 
  /*                       } */
  /*                   } */
  /*               } */
  /*           } */
  /*           for (int ii = i; ii < i_max; ++ii) { */
  /*               const int row = ii*p; */
  /*               for (int jj = j; jj < j_max; ++jj) { */
  /*                   C[row + jj].mask_and_send_dot(); */
  /*               } */
  /*           } */
  /*       } */
  /*   } */
        




        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward21(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.transpose().data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    for (int jj = j; jj < j_max; ++jj) {
                        for (int kk = k; kk < k_max; ++kk) {
                       C[ii*p + jj] += A[ii*f+kk] * B[jj*f + kk]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward22(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    MatX<T> BM = im_col.transpose();
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    for (int jj = j; jj < j_max; ++jj) {
                        T temp_sum = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                       temp_sum += A[ii*f+kk] * B[jj*f + kk]; 
                        }
                        C[ii*p + jj] += temp_sum;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward23(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    MatX<T> BM = im_col.transpose();
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                    for (int jj = j; jj < j_max; ++jj) {
                for (int ii = i; ii < i_max; ++ii) {
                    int row = ii*f+k;
                    int row2 = jj*f+k;
                    /* const int row2 = ii*f+kk; */
                        T temp_sum = T(0);
                        while(row < ii*f+k_max){
                       temp_sum += A[row++] * B[row2++]; 
                        }
                        C[ii*p + jj] += temp_sum;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                int row = ii*p+j;
                while(row < ii*p+j_max){
                    C[row++].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward24(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    MatX<T> BM = im_col.transpose();
    auto B = BM.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                    for (int jj = j; jj < j_max; ++jj) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                        T temp_sum = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                       temp_sum += A[ii*f+kk] * B[jj*f + kk]; 
                        }
                        C[ii*p + jj] += temp_sum;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                for (int jj = j; jj < j_max; ++jj) {
                    C[ii*p + jj].mask_and_send_dot();
                }
            }
        }
    }
        

        }


        
    
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward25(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            C[i*p + j] = T(0);
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward26(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
  for (int i = 0; i < m; i += TILE_SIZE) {
      C[i*p + j] = T(0);
        int i_max = std::min(i + TILE_SIZE, m);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    template<typename T>
	void Conv2d<T>::forward27(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
        for( int i = 0; i < this->output.size(); ++i) {
            this->output(i) = T(0);
        }
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
        }
    }

            for (int i = 0; i < m*p; ++i) {
                this->output(i).mask_and_send_dot();
            }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    template<typename T>
	void Conv2d<T>::forward28(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
        for( int i = 0; i < this->output.size(); ++i) {
            this->output(i) = T(0);
        }
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
                        for (int kk = k; kk < k_max; ++kk) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
        }
    }

            for (int i = 0; i < m*p; ++i) {
                this->output(i).mask_and_send_dot();
            }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    template<typename T>
	void Conv2d<T>::forward29(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
        for( int i = 0; i < this->output.size(); ++i) {
            this->output(i) = T(0);
        }
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
                for (int ii = i; ii < i_max; ++ii) {
                        for (int kk = k; kk < k_max; ++kk) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = A[ii*f+kk];
                    for (int jj = j; jj < j_max; ++jj) {
                       C[ii*p + jj] += temp * B[kk*p + jj]; 
                        }
                    }
                }
            }
        }
    }

            for (int i = 0; i < m*p; ++i) {
                this->output(i).mask_and_send_dot();
            }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }

    template<typename T>
	void Conv2d<T>::forward30(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
        for( int i = 0; i < this->output.size(); ++i) {
            this->output(i) = T(0);
        }
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    for (int jj = j; jj < j_max; ++jj) {
                    auto temp = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                       temp += A[ii*f+kk] * B[kk*p + jj]; 
                        }
                    C[ii*p + jj] += temp;
                    }
                }
            }
        }
    }

            for (int i = 0; i < m*p; ++i) {
                this->output(i).mask_and_send_dot();
            }
        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }
    
    template<typename T>
	void Conv2d<T>::forward31(const MatX<T>& prev_out, bool is_training)
	{
            /* for(int i = 0; i < oc; ++i) { */
            /*     T sum = T(0); */
            /*     for(int j = 0; j < ohw; ++j) { */
            /*             for(int k = 0; k < kernel.cols(); ++k) { */
            /*         sum += (kernel(i, k) * im_col(k, j));  // Use custom * and + operators */
        std::cout << "prev_out: " << prev_out.size() << std::endl;
        std::cout << "output: " << this->output.size() << std::endl;
        for( int i = 0; i < this->output.size(); ++i) {
            this->output(i) = T(0);
        }
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
            /* auto A = kernel; */
            /* auto B = im_col; */
            /* auto C = this->output; */

            auto A = kernel.data();
    auto B = im_col.data();
    auto C = this->output.data() + (oc * ohw) * n;
    
    const int m = oc;
    const int f = kernel.cols();
    const int p = ohw;
  for (int i = 0; i < m; i += TILE_SIZE) {
        int i_max = std::min(i + TILE_SIZE, m);
        for (int j = 0; j < p; j += TILE_SIZE) {
            int j_max = std::min(j + TILE_SIZE, p);
            for (int k = 0; k < f; k += TILE_SIZE) {
                int k_max = std::min(k + TILE_SIZE, f);
                    for (int jj = j; jj < j_max; ++jj) {
                for (int ii = i; ii < i_max; ++ii) {
                    /* const int row2 = ii*f+kk; */
                    auto temp = T(0);
                        for (int kk = k; kk < k_max; ++kk) {
                       temp += A[ii*f+kk] * B[kk*p + jj]; 
                        }
                    C[ii*p + jj] += temp;
                    }
                }
            }
            for (int ii = i; ii < i_max; ++ii) {
                const int row = ii*p;
                for (int jj = j; jj < j_max; ++jj) {
                    C[row + jj].mask_and_send_dot();
                }
            }
        }
    }

        




        } 
        /* for (int i = 0; i < this->output.size(); i++) { */
        /*     this->output(i).mask_and_send_dot(); */
    /* } */
            T::communicate();
            for (int i = 0; i < this->output.size(); i++) {
                this->output(i).complete_mult();
            }
		for (int n = 0; n < batch; n++) {
			this->output.block(oc * n, 0, oc, ohw).colwise() += bias;
	}
    }


    template<typename T>
	void Conv2d<T>::backward(const MatX<T>& prev_out, MatX<T>& prev_delta)
	{
		for (int n = 0; n < batch; n++) {
			const T* im = prev_out.data() + (ic * ihw) * n;
			im2col(im, ic, ih, iw, kh, 1, pad, im_col.data());
			dkernel += this->delta.block(oc * n, 0, oc, ohw) * im_col.transpose();
			dbias += this->delta.block(oc * n, 0, oc, ohw).rowwise().sum();
		}

		if (!this->is_first) {
			for (int n = 0; n < batch; n++) {
				T* begin = prev_delta.data() + ic * ihw * n;
				im_col = kernel.transpose() * this->delta.block(oc * n, 0, oc, ohw);
				col2im(im_col.data(), ic, ih, iw, kh, 1, pad, begin);
			}
		}
	}

template <typename T>
class SH{
DATATYPE s1;
DATATYPE s2;
    public:



static SH get_S(UINT_TYPE val){
SH s;
UINT_TYPE s_arr[sizeof(T)/sizeof(UINT_TYPE)] = {val};
UINT_TYPE r_arr[sizeof(T)/sizeof(UINT_TYPE)];
for(int i = 0; i < sizeof(T)/sizeof(UINT_TYPE); i++){
    r_arr[i] = rand();
}
orthogonalize_arithmetic(s_arr, &s.s1 , 1);
orthogonalize_arithmetic(r_arr, &s.s2 , 1);
s.s1 = OP_SUB(s.s1, s.s2);
return s;
}

SH(UINT_TYPE val){
UINT_TYPE s_arr[sizeof(T)/sizeof(UINT_TYPE)] = {val};
orthogonalize_arithmetic(s_arr, &s1 , 1);
s2 = SET_ALL_ZERO();
}



SH(T s1, T s2){
this->s1 = s1;
this->s2 = s2;
}


SH(){
this->s1 = SET_ALL_ZERO();
this->s2 = SET_ALL_ZERO();
}


SH operator+(const SH s) const{
    return SH(this->s1 + s.s1, this->s2 + s.s2);
}

SH operator-(const SH s) const{
    return SH(this->s1 - s.s1, this->s2 - s.s2);
}

SH operator*(const SH s) const{
    auto ls1 = OP_ADD( OP_MULT(this->s1, s.s1), OP_MULT(this->s1, s.s2));
    auto ls2 = OP_ADD( OP_MULT(this->s2, s.s1), OP_MULT(this->s2, s.s2));
    return SH(ls1, ls2);
}

SH operator*(const UINT_TYPE s) const{
    return SH(OP_MULT(s1, PROMOTE(s)), OP_MULT(s2, PROMOTE(s)));
}

SH operator/(const UINT_TYPE s) const{
    /* return SH(OP_DIV(s1, PROMOTE(s)), OP_DIV(s2, PROMOTE(s))); */ // not supported for now
    return SH();
}

void operator+=(const SH s){
    this->s1 += s.s1;
    this->s2 += s.s2;
}

void operator-=(const SH s){
    this->s1 -= s.s1;
    this->s2 -= s.s2;
}

void operator*= (const SH s){
    this->s1 = OP_ADD( OP_MULT(this->s1, s.s1), OP_MULT(this->s1, s.s2));
    this->s2 = OP_ADD( OP_MULT(this->s2, s.s1), OP_MULT(this->s2, s.s2));
}


//needed for Eigen optimization
bool operator==(const SH& other) const {
    return false; 
}

SH trunc_local() const{
    return SH(OP_TRUNC(s1), OP_TRUNC(s2));
}

template<typename float_type, int fractional>
float_type reveal_float() const{

    UINT_TYPE s_arr[sizeof(T)/sizeof(UINT_TYPE)];
    T temp = OP_ADD(s1, s2);
    unorthogonalize_arithmetic(&temp, s_arr, 1);
    float_type result = fixedToFloat<float_type, UINT_TYPE, fractional>(s_arr[0]);
    return result;
    }


};

template<typename T>
SH<T> truncate(const SH<T>& val) {
    return val.trunc_local();
}




    template<typename Share>
void forward_pass(DATATYPE* res)
{
Share::communicate(); // Dummy communication round to simulate input sharing
using D = sint_t<Matrix_Share<DATATYPE, Share>>;
/* using D = Matrix_Share<DATATYPE, Share>; */
/* using M = SH<DATATYPE>; */
/* using D = SH<DATATYPE>; */
/* Conv2d<M> conv(3,64,3,1); */
#if FUNCTION_IDENTIFIER == 20
std::vector<int> input_shape = {1, 3, NUM_INPUTS, NUM_INPUTS};
MatX<D> input(1, NUM_INPUTS * NUM_INPUTS * 3);
Conv2d<D> d_conv(3, 64, 3, 1, "xavier_normal");
#elif FUNCTION_IDENTIFIER == 23
Conv2d<D> d_conv(64, 64, 3, 1, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method)
vector<int> input_shape = {1, 64, NUM_INPUTS, NUM_INPUTS};
MatX<D> input(1, 64 * NUM_INPUTS * NUM_INPUTS);
#else
Conv2d<D> d_conv(64, 128, 3, 1, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method)
vector<int> input_shape = {1, 64, NUM_INPUTS/2, NUM_INPUTS/2};
MatX<D> input(1, 64 * NUM_INPUTS/2 * NUM_INPUTS/2);
#endif
    d_conv.set_layer(input_shape);
    for (int j = 0; j < d_conv.output.size(); j++) {
    d_conv.output(j).mask_and_send_dot();
    }
    Share::communicate();
    for (int j = 0; j < d_conv.output.size(); j++) {
    d_conv.output(j).complete_mult();
    }

    //dummy reveal
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    UINT_TYPE dummy[DATTYPE];
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);


}
template<typename Share>
void cryptgpu_figure1a(DATATYPE* res)
{
using D = Matrix_Share<DATATYPE, Share>;
const int batch = 1;
Conv2d<D> d_conv(3, 64, 11, 4, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method)
vector<int> input_shape = {batch, 64, NUM_INPUTS, NUM_INPUTS};
MatX<D> input(batch, 64 * NUM_INPUTS * NUM_INPUTS);

}
    
    template<typename Share>
void conv2D_bench(DATATYPE* res)
{
Share::communicate(); // Dummy communication round to simulate input sharing
/* using D = Additive_Share<DATATYPE, Share>; */
using D = Matrix_Share<DATATYPE, Share>;
const int batch = 1;
/* using D = Matrix_Share<DATATYPE, Share>; */
/* using M = SH<DATATYPE>; */
/* using D = SH<DATATYPE>; */
/* Conv2d<M> conv(3,64,3,1); */
Conv2d<D> d_conv(64, 64, 3, 1, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method)
vector<int> input_shape = {batch, 64, NUM_INPUTS, NUM_INPUTS};
MatX<D> input(batch, 64 * NUM_INPUTS * NUM_INPUTS);
    d_conv.set_layer(input_shape);
    alignas(sizeof(DATATYPE))UINT_TYPE dummy[DATTYPE];
#if FUNCTION_IDENTIFIER == 56
d_conv.forward4(input, false);
#elif FUNCTION_IDENTIFIER == 57
d_conv.forward20(input, false);
#elif FUNCTION_IDENTIFIER == 37
d_conv.forward1(input, false);
#elif FUNCTION_IDENTIFIER == 38
d_conv.forward10(input, false);
#elif FUNCTION_IDENTIFIER == 39
d_conv.forward11(input, false);
#elif FUNCTION_IDENTIFIER == 40
d_conv.forward17(input, false);
#elif FUNCTION_IDENTIFIER == 41
d_conv.forward18(input, false);
#elif FUNCTION_IDENTIFIER == 42
d_conv.forward21(input, false);
#elif FUNCTION_IDENTIFIER == 43
d_conv.forward26(input, false);
#elif FUNCTION_IDENTIFIER == 44
d_conv.forward21(input, false);
#elif FUNCTION_IDENTIFIER == 45
d_conv.forward9(input, false);
#elif FUNCTION_IDENTIFIER == 46
d_conv.forward10(input, false);
#elif FUNCTION_IDENTIFIER == 47
d_conv.forward11(input, false);
#elif FUNCTION_IDENTIFIER == 48
d_conv.forward12(input, false);
#elif FUNCTION_IDENTIFIER == 49
d_conv.forward13(input, false);
#elif FUNCTION_IDENTIFIER == 50
d_conv.forward14(input, false);
#elif FUNCTION_IDENTIFIER == 51
d_conv.forward23(input, false);
#elif FUNCTION_IDENTIFIER == 52
d_conv.forward16(input, false);
#elif FUNCTION_IDENTIFIER == 53
d_conv.forward17(input, false);
#elif FUNCTION_IDENTIFIER == 54
std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward10(input, false);
std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 10: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward1(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 1: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward2(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 2: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward3(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 3: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward4(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 4: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward5(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 5: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward6(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 6: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward7(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 7: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward8(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 8: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward9(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken 9: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward11(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 11: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward12(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 12: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward13(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 13: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward14(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 14: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward15(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 15: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward16(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 16: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward17(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 17: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward18(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 18: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward19(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 19: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward20(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 20: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward21(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken 21: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward22(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 22: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward23(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 23: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward24(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 24: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward25(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 25: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward26(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 26: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward27(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 27: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward28(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 28: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward29(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 29: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward30(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 30: " << duration << std::endl;
t1 = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 5; i++) 
d_conv.forward31(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
std::cout << "Time taken 31: " << duration << std::endl;
#elif FUNCTION_IDENTIFIER == 55
TILE_SIZE = 8;
std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward17(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 17, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 17: " << duration << std::endl;


t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward18(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 18, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 18: " << duration << std::endl;

t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward18_old(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 18 (old), TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 18: " << duration << std::endl;



t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward11(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 11, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;


t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward12(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 12, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 12: " << duration << std::endl;


t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 11: " << duration << std::endl;


t1 = std::chrono::high_resolution_clock::now();
while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward10(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 10, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 10: " << duration << std::endl;

while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward12(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 12, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 12: " << duration << std::endl;

while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward13(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 13, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 13: " << duration << std::endl;

while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward15(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 15, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 15: " << duration << std::endl;

while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward16(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 16, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 16: " << duration << std::endl;

while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward21(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 21, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 21: " << duration << std::endl;


while(TILE_SIZE < 225)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    d_conv.forward26(input, false);
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    std::cout << "Time taken for 26, TILE_SIZE: " << TILE_SIZE << " " << duration << std::endl;
    TILE_SIZE += 8;
}
TILE_SIZE = 8;
t2 = std::chrono::high_resolution_clock::now();
duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
std::cout << "Time taken for 26: " << duration << std::endl;


#endif
    //dummy reveal
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);


}


template<typename Share>
void backward_pass(DATATYPE* res)
{

Share::communicate(); // Dummy communication round to simulate input sharing
using D = sint_t<Matrix_Share<DATATYPE, Share>>;
/* using D = Matrix_Share<DATATYPE, Share>; */
#if FUNCTION_IDENTIFIER == 21 
std::vector<int> input_shape = {1, 3, NUM_INPUTS, NUM_INPUTS};
MatX<D> input(1, NUM_INPUTS * NUM_INPUTS * 3);
Conv2d<D> d_conv(3, 64, 3, 1, "xavier_normal");
#elif FUNCTION_IDENTIFIER == 24
Conv2d<D> d_conv(64, 64, 3, 1, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method)
vector<int> input_shape = {1, 64, NUM_INPUTS, NUM_INPUTS};
MatX<D> input(1, 64 * NUM_INPUTS * NUM_INPUTS);
#else
Conv2d<D> d_conv(64, 128, 3, 1, "xavier_normal"); // Assuming Conv2d takes in(input_channels, output_channels, kernel_size, stride, initialization_method)
vector<int> input_shape = {1, 64, NUM_INPUTS/2, NUM_INPUTS/2};
MatX<D> input(1, 64 * NUM_INPUTS/2 * NUM_INPUTS/2);
#endif
d_conv.set_layer(input_shape);

/* conv.set_layer(input_shape); */
d_conv.backward(input,d_conv.output);
    for (int j = 0; j < d_conv.output.size(); j++) {
    d_conv.output(j).mask_and_send_dot();
    }
    Share::communicate();
    for (int j = 0; j < d_conv.output.size(); j++) {
    d_conv.output(j).complete_mult();
    }
    
    //dummy reveal
    d_conv.output(d_conv.output.size() - 1).prepare_reveal_to_all();
    Share::communicate();
    UINT_TYPE dummy[DATTYPE];
    d_conv.output(d_conv.output.size() - 1).complete_reveal_to_all(dummy);

}


    template<typename Share>
void FC_bench(DATATYPE* res)
{
    Share::communicate(); // Dummy communication round to simulate input sharing
    using S = sint_t<Matrix_Share<DATATYPE, Share>>;
    /* using M = Matrix_Share<DATATYPE, Share>; */
    VecX<S> a(NUM_INPUTS);
    VecX<S> c(NUM_INPUTS);
    MatX<S> b(NUM_INPUTS, NUM_INPUTS);
    c = b * a;
    
    for(int i = 0; i < NUM_INPUTS; i++)
    {
            c(i).mask_and_send_dot();
    }

    Share::communicate();
    
    for(int i = 0; i < NUM_INPUTS; i++)
    {
            c(i).complete_mult();
    }
    
    //dummy reveal
    c(NUM_INPUTS - 1).prepare_reveal_to_all();
    Share::communicate();
    UINT_TYPE dummy[DATTYPE];
    c(NUM_INPUTS - 1).complete_reveal_to_all(dummy);
}

template<typename Share>
void dot_prod_eigen_bench(DATATYPE* res)
{
    Share::communicate(); // Dummy communication round to simulate input sharing
    using S = Matrix_Share<DATATYPE, Share>;
    VecX<S> a(NUM_INPUTS);
    VecX<S> c(NUM_INPUTS);
    MatX<S> b(NUM_INPUTS, NUM_INPUTS);
    c = b * a;
    
    for(int i = 0; i < NUM_INPUTS; i++)
    {
            c(i).mask_and_send_dot();
    }

    Share::communicate();
    
    for(int i = 0; i < NUM_INPUTS; i++)
    {
            c(i).complete_mult();
    }

    //dummy reveal
    c(NUM_INPUTS - 1).prepare_reveal_to_all();
    Share::communicate();
    UINT_TYPE dummy[DATTYPE];
    c(NUM_INPUTS - 1).complete_reveal_to_all(dummy);
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
