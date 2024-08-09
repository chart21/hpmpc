#pragma once
#include "../../config.h"
#include "../../datatypes/float_fixed_converter.hpp"

    template<typename T>                    
void prepare_prob_div(T &out, const int denominator)
{
    #if TRUNC_APPROACH == 0
    if ((denominator & (denominator - 1)) == 0) // if power of 2
            out = out.prepare_div_exp2(denominator);
    else
        out *= FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/float(denominator)); 
    #else
        /* #if TRUNC_THEN_MULT == 0 */
        out = out.mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/float(denominator))); 
        /* #else */
        /*     return; */
        /* #endif */
    #endif
}

template<typename T>
void complete_prob_div(T &out, const int len, const int denominator)
{

#if TRUNC_APPROACH == 0
        for (int i = 0; i < len; i++)
            out[i].complete_public_mult_fixed();
#else
    /* #if TRUNC_THEN_MULT == 1 */
    /*     for (int i = 0; i < len; i++) */
    /*         out[i] = out[i].mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/float(denominator))); */
    /* #endif */
    #if TRUNC_APPROACH == 1
        trunc_2k_in_place(out, len, false);
#elif TRUNC_APPROACH == 2
        trunc_exact_in_place(out, len);
    #endif
#endif
}


