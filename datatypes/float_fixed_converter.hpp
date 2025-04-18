#pragma once
#include "../config.h"
#include "../core/include/pch.h"
#define DEBUG_MODE
template <typename float_type, typename INT_TYPE, typename UINT_TYPE, int fractional_bits>
struct FloatFixedConverter
{
    static float_type fixed_to_float(INT_TYPE fixed_val, int frac_bits = fractional_bits)
    {
#if TRUNC_THEN_MULT == 1
        const float_type scale = (UINT_TYPE(1) << frac_bits * 2);
#else
        const float_type scale = (UINT_TYPE(1) << frac_bits);
#endif
        return static_cast<float_type>(fixed_val) / scale;
    }

    static INT_TYPE float_to_fixed(float_type float_val, int frac_bits = fractional_bits)
    {
#if TRUNC_THEN_MULT == 1
        const float_type scale = (UINT_TYPE(1) << frac_bits * 2);
        float_val = float_val / (UINT_TYPE(1) << frac_bits);  // variant with trunc then mult
#else
        const float_type scale = (UINT_TYPE(1) << frac_bits);
#endif
        // Check for overflow and underflow
        if (float_val >= (std::numeric_limits<INT_TYPE>::max()) / scale)
        {  // Modified check
#if PRINT_IMPORTANT == 1
            std::cout << "Warning: Overflow occurred! -> clamping" << float_val << std::endl;
#endif
            return std::numeric_limits<INT_TYPE>::max();
        }

        if (float_val <= std::numeric_limits<INT_TYPE>::min() / scale)
        {
#if PRINT_IMPORTANT == 1
            std::cout << "Warning: Underflow occurred! -> clamping" << std::endl;
#endif
            return std::numeric_limits<INT_TYPE>::min();
        }

        return static_cast<INT_TYPE>(std::round(float_val * scale));
    }

    static UINT_TYPE int_to_twos_complement(INT_TYPE val) { return static_cast<UINT_TYPE>(val); }

    static INT_TYPE twos_complement_to_int(UINT_TYPE val) { return static_cast<INT_TYPE>(val); }

    static UINT_TYPE float_to_ufixed(float_type float_val, int frac_bits = fractional_bits)
    {
        return int_to_twos_complement(float_to_fixed(float_val, frac_bits));
    }

    static float_type ufixed_to_float(UINT_TYPE ufixed_val, int frac_bits = fractional_bits)
    {
        return fixed_to_float(twos_complement_to_int(ufixed_val), frac_bits);
    }
};

template <typename float_type, typename INT_TYPE, typename UINT_TYPE, int fractional>
float_type fixedToFloat(UINT_TYPE val, int frac_bits = fractional)
{
    return FloatFixedConverter<float_type, INT_TYPE, UINT_TYPE, fractional>::ufixed_to_float(val, frac_bits);
}

template <typename float_type, typename INT_TYPE, typename UINT_TYPE, int fractional>
UINT_TYPE floatToFixed(float_type val, int frac_bits = fractional)
{
    return FloatFixedConverter<float_type, INT_TYPE, UINT_TYPE, fractional>::float_to_ufixed(val, frac_bits);
}
