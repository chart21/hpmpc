#pragma once
#include "../../config.h"
#include "../../datatypes/float_fixed_converter.hpp"

template <typename T>
void prepare_prob_div(T& out, const int denominator, const int frac_bits = FRACTIONAL)
{
#if TRUNC_APPROACH == 0 || TRUNC_APPROACH == 4
#if TRUNC_APPROACH == 0
    out =
        out.prepare_mult_public_fixed(FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(
                                          1 / FLOATTYPE(denominator), frac_bits),
                                      frac_bits);
#elif TRUNC_APPROACH == 4
    /* if(frac_bits <= FRACTIONAL / 2) */
    out =
        out.prepare_mult_public_fixed(FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(
                                          1 / FLOATTYPE(denominator), frac_bits),
                                      frac_bits);
    /* else */
    /* out = out.mult_public(FloatFixedConverter<float, INT_TYPE, UINT_TYPE,
     * FRACTIONAL>::float_to_ufixed(1/float(denominator),frac_bits)); */
#endif
#else
    out = out.mult_public(FloatFixedConverter<FLOATTYPE, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(
        1 / FLOATTYPE(denominator), frac_bits));
#endif
}

template <typename T>
void complete_prob_div(T& out, const int len, const int denominator, const int frac_bits = FRACTIONAL)
{

#if TRUNC_APPROACH == 0 || TRUNC_APPROACH == 4
#if TRUNC_APPROACH == 0
    for (int i = 0; i < len; i++)
        out[i].complete_public_mult_fixed();
#elif TRUNC_APPROACH == 4
    /* if (frac_bits <= FRACTIONAL / 2) */
    /* { */
    for (int i = 0; i < len; i++)
        out[i].complete_public_mult_fixed();
        /* } */
        /*     else */
        /*     trunc_2k_in_place(out, len, true, frac_bits); */
#endif
#else
#if TRUNC_APPROACH == 1
    trunc_2k_in_place(out, len, all_positive, frac_bits);
#elif TRUNC_APPROACH == 2
    trunc_exact_in_place(out, len, frac_bits);
#elif TRUNC_APPROACH == 3
    trunc_exact_opt_in_place(out, len, all_positive, frac_bits);
#endif
#endif
}
