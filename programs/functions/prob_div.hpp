#pragma once
#include "../../config.h"
#include "../../datatypes/float_fixed_converter.hpp"

    template<typename T>                    
void prepare_prob_div(T &out, const int denominator)
{
    #if TRUNC_APPROACH == 0 || TRUNC_APPROACH == 4
    if ((denominator & (denominator - 1)) == 0) // if power of 2
            out = out.prepare_div_exp2(denominator);
    else
#if TRUNC_APPROACH == 0
        out *= FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/float(denominator)); 
#elif TRUNC_APPROACH == 4
        out = out.mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/float(denominator))); 
#endif
    #else
        out = out.mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(1/float(denominator))); 
    #endif
}

template<typename T>
void complete_prob_div(T &out, const int len, const int denominator)
{

#if TRUNC_APPROACH == 0 || TRUNC_APPROACH == 4
#if TRUNC_APPROACH == 0     
    for (int i = 0; i < len; i++)
            out[i].complete_public_mult_fixed();
#elif TRUNC_APPROACH == 4
    if ((denominator & (denominator - 1)) == 0) // if power of 2
    { 
        for (int i = 0; i < len; i++)
                out[i].complete_public_mult_fixed();
    } 
        else
        trunc_2k_in_place(out, len, true);
#endif
#else
    #if TRUNC_APPROACH == 1
        trunc_2k_in_place(out, len, true);
#elif TRUNC_APPROACH == 2
        trunc_exact_in_place(out, len);
#elif TRUNC_APPROACH == 3
        trunc_exact_opt_in_place(out, len, true);
    #endif
#endif
}


