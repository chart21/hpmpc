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

#include "../../utils/print.hpp"

#include <cmath>
#include <eigen3/Eigen/Core>

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
#elif FUNCTION_IDENTIFIER == 21 || FUNCTION_IDENTIFIER == 24 || FUNCTION_IDENTIFIER == 26
#define FUNCTION backward_pass
#elif FUNCTION_IDENTIFIER == 22
#define FUNCTION FC_bench
#elif FUNCTION_IDENTIFIER == 27
#define FUNCTION dot_prod_eigen_bench
#elif FUNCTION_IDENTIFIER == 28
#define FUNCTION argmax_test
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

    template<typename Share>
void fixed_test(DATATYPE* res)
{
    using M = Matrix_Share<DATATYPE, Share>;
    using sint = sint_t<M>;
    auto a = new sint[NUM_INPUTS][NUM_INPUTS];
    auto b = new sint[NUM_INPUTS][NUM_INPUTS];
    auto c = new sint[NUM_INPUTS][NUM_INPUTS];
    std::memset(c, 0, sizeof(M) * NUM_INPUTS * NUM_INPUTS);
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            a[i][j]. template prepare_receive_from<P_0>();
            b[i][j]. template prepare_receive_from<P_1>();
        }
}
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            a[i][j]. template complete_receive_from<P_0>();
            b[i][j]. template complete_receive_from<P_1>();
        }
    }


    Share::communicate(); // dummy round
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            for(int k = 0; k < NUM_INPUTS; k++)
            {
                c[i][j] = c[i][j] + a[i][k] * b[k][j];
                /* c[i][j].prepare_dot_add(a[i][k], b[k][j]); */
            }
                c[i][j].mask_and_send_dot();
        }
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            c[i][j].complete_mult();
        }
}
    Share::communicate();
delete[] a;
delete[] b;

    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            c[i][j].prepare_reveal_to_all();
        }
    }
    auto result_arr = new UINT_TYPE[NUM_INPUTS*2][DATTYPE];
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            c[i][j].complete_reveal_to_all(result_arr[2*i+j]);
            /* if(current_phase == 1) */
            /* { */
            /* #if FRACTIONAL > 0 */
            /* std::cout << fixedToFloat<float, UINT_TYPE, FRACTIONAL>(result_arr[i][j][0]) << std::endl; */
            /* #else */
            /* std::cout << result_arr[2+i+j][0] << std::endl; */
            /* #endif */
            }
        }
            if(current_phase == 1)
    {
        std::cout << "P" << PARTY << ": Result: ";
        for (int i = 0; i < 4; i++)
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
    /* using Bitset = sbitset_t<S>; */
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
    using Bitset = sbitset_t<S>;
    using sint = sint_t<A>;
    
    sint* val = new sint[NUM_INPUTS];
    Bitset *s1 = new Bitset[NUM_INPUTS];
    Bitset *s2 = new Bitset[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        s1[i] = sbitset_t<S>::prepare_A2B_S1( (S*) val[i].get_share_pointer());
        s2[i] = sbitset_t<S>::prepare_A2B_S2( (S*) val[i].get_share_pointer());
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
    std::vector<PPA_MSB_Unsafe<S>> adders;
    #else
    std::vector<BooleanAdder_MSB<S>> adders;
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
/* using Bitset = sbitset_t<S>; */
/* using sint = sint_t<A>; */
    /* sint* val = new sint; */
    /* Bitset *s1 = new Bitset; */
    /* Bitset *s2 = new Bitset; */
    /* for(int i = 0; i < NUM_INPUTS; i++) */
    /* { */
    /*     s1[i] = sbitset_t<S>::prepare_A2B_S1( (S*) val[i].get_share_pointer()); */
    /*     s2[i] = sbitset_t<S>::prepare_A2B_S2( (S*) val[i].get_share_pointer()); */
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
using Bitset = sbitset_t<S>;
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
template<typename Datatype, typename Share>
void get_msb_range(sint_t<Additive_Share<Datatype, Share>>* val, XOR_Share<Datatype, Share>* msb, int len)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using Bitset = sbitset_t<S>;
using sint = sint_t<A>;
    Bitset *s1 = new Bitset[len];
Bitset *s2 = new Bitset[len];
    for(int i = 0; i < len; i++)
    {
        s1[i] = sbitset_t<S>::prepare_A2B_S1( (S*) val[i].get_share_pointer());
        s2[i] = sbitset_t<S>::prepare_A2B_S2( (S*) val[i].get_share_pointer());
    }
    Share::communicate();
    for(int i = 0; i < len; i++)
    {
        s1[i].complete_A2B_S1();
        s2[i].complete_A2B_S2();
    }
    /* Bitset* y = new Bitset[NUM_INPUTS]; */

    std::vector<BooleanAdder_MSB<S>> adders;
    
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

template<typename Datatype, typename Share>
void max_min_msb_range(sint_t<Additive_Share<Datatype, Share>>* val, XOR_Share<Datatype, Share>* msb, int m, bool want_max)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using Bitset = sbitset_t<S>;
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

    get_msb_range(max_val, msb, counter);
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
    
    template<typename Datatype, typename Share>
sint_t<Additive_Share<Datatype, Share>> max_min(sint_t<Additive_Share<Datatype, Share>>* begin, sint_t<Additive_Share<Datatype, Share>>* end, bool want_max)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using Bitset = sbitset_t<S>;
using sint = sint_t<A>;
   int m = end - begin;
   int og_len = m;

   sint* val = new sint[m];
    std::copy(begin, end, val);
   if(m == 1)
   {
       return val[0];
   }

   int log2m = std::ceil(std::log2(m)); 
   for(int i = 0; i < log2m; i++)
   {
       int counter = m/2; // 
        int offset = m % 2; // if m is odd, offset is 1
        S* msb = new S[m];
        if(want_max)
            max_min_msb_range(val,msb,m,true); //get msb and max of 0 -> counter
        else
            max_min_msb_range(val,msb,m,false); //get msb and max of 0 -> counter

        delete[] msb;
        m = counter + offset;
    }
   return val[0];
}


    template<typename Datatype, typename Share>
void argmax_argmin(sint_t<Additive_Share<Datatype, Share>>* begin, sint_t<Additive_Share<Datatype, Share>>* end, XOR_Share<Datatype, Share>* output, bool want_max)
{
using S = XOR_Share<Datatype, Share>;
using A = Additive_Share<Datatype, Share>;
using Bitset = sbitset_t<S>;
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
            max_min_msb_range(val,msb,m,true); //get msb and max of 0 -> counter
        else
            max_min_msb_range(val,msb,m,false); //get msb and max of 0 -> counter

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
using Bitset = sbitset_t<S>;
using sint = sint_t<A>;
auto a = new sint[NUM_INPUTS];
auto max_output = new S[NUM_INPUTS];
auto min_output = new S[NUM_INPUTS];
for(int i = 0; i < NUM_INPUTS; i++)
        a[i]. template prepare_receive_from<P_0>();
Share::communicate();
for(int i = 0; i < NUM_INPUTS; i++)
        a[i]. template complete_receive_from<P_0>();
Share::communicate();
                /* #if PARTY == 2 */
        /* for(int i = 0; i < NUM_INPUTS; i++) */
            /* std::cout << "a: " << a[i].get_p1() << std::endl; */
        /* #endif */ 
argmax_argmin(a, a+NUM_INPUTS, max_output,true);
argmax_argmin(a, a+NUM_INPUTS, min_output,false);
auto max_val = max_min(a, a+NUM_INPUTS, true);
auto min_val = max_min(a, a+NUM_INPUTS, false);
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
auto max_int = new UINT_TYPE[BITLENGTH * sizeof(DATATYPE)/sizeof(UINT_TYPE)];
auto min_int = new UINT_TYPE[BITLENGTH * sizeof(DATATYPE)/sizeof(UINT_TYPE)];
max_val.complete_reveal_to_all(max_int);
min_val.complete_reveal_to_all(min_int);
if(current_phase == 1)
{
for(int i = 0; i < NUM_INPUTS; i++)
    std::cout << "arg_max: " << "Index: " << i << " Value: " << result_arr[0][i] << std::endl;
for(int i = 0; i < NUM_INPUTS; i++)
    std::cout << "arg_min: " << "Index: " << i << " Value: "<< result_arr[1][i] << std::endl;
std::cout << "max: " << max_int[0] << std::endl;
std::cout << "min: " << int(min_int[0]) << std::endl;
}
delete[] a;
delete[] max_output;
delete[] min_output;
delete[] result_arr;
}


#endif




#if FUNCTION_IDENTIFIER > 19

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
