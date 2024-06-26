#pragma once
#include "../../config.h"
#include "truncation.hpp"
#include "../../datatypes/float_fixed_converter.hpp"

    template<typename T>                    
void prepare_prob_div(T &out, const int denominator)
{
    if ((denominator & (denominator - 1)) == 0) // if power of 2
    #if TRUNC_APPROACH == 0
            out = out.prepare_div_exp2(denominator);
    #else
        #if TRUNC_THEN_MULT == 0
        out = out.mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/flaot(denominator))); 
        #else
            continue;
        #endif
    #endif
    else 
    #if TRUNC_APPROACH == 0
        out *= FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/denominator); 
    #else
        #if TRUNC_THEN_MULT == 0
        out = out.mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/float(denominator))); 
        #else
            continue;
        #endif
    #endif
}

template<typename T>
void complete_prob_div(T &out, const int len, const int denominator)
{

#if TRUNC_APPROACH == 0
        for (int i = 0; i < len; i++)
            out[i].complete_public_mult_fixed();
#else
    #if TRUNC_THEN_MULT == 1
        trunc_2k_in_place(out, len);
        T::communicate();
        for (int i = 0; i < len; i++)
            out[i] = out[i].mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/float(denominator)));
    #else
        trunc_2k_in_place(out, len);
        T::communicate();
    #endif
#endif
}


