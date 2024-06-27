#pragma once
#include "../core/include/pch.h"
#include "../config.h"
#define DEBUG_MODE
template <typename float_type, typename INT_TYPE, typename UINT_TYPE, int fractional_bits>
struct FloatFixedConverter {
static float_type fixed_to_float(INT_TYPE fixed_val) {
#if TRUNC_THEN_MULT == 1
    const float_type scale = (1 << fractional_bits*2);
#else
    const float_type scale = (1 << fractional_bits);
#endif
    return static_cast<float_type>(fixed_val) / scale;
}

static INT_TYPE float_to_fixed(float_type float_val) {
#if TRUNC_THEN_MULT == 1
    const float_type scale = (1 << fractional_bits*2);
    float_val = float_val/ (1 << fractional_bits); // variant with trunc then mult
#else
    const float_type scale = (1 << fractional_bits);
#endif
  // Check for overflow and underflow
    if (float_val >= (std::numeric_limits<INT_TYPE>::max()) / scale) { // Modified check
#if PRINT_IMPORTANT == 1
        std::cout << "Warning: Overflow occurred! -> clamping" << float_val << std::endl;
#endif
        return std::numeric_limits<INT_TYPE>::max();
    }

    if (float_val <= std::numeric_limits<INT_TYPE>::min() / scale) {
#if PRINT_IMPORTANT == 1
        std::cout << "Warning: Underflow occurred! -> clamping" << std::endl;
#endif
        return std::numeric_limits<INT_TYPE>::min();
    }

    return static_cast<INT_TYPE>(std::round(float_val * scale));
}

static UINT_TYPE int_to_twos_complement(INT_TYPE val) {
    return static_cast<UINT_TYPE>(val); 
}

static INT_TYPE twos_complement_to_int(UINT_TYPE val) {
    return static_cast<INT_TYPE>(val);
}

static UINT_TYPE float_to_ufixed(float_type float_val) { 
    return int_to_twos_complement(float_to_fixed(float_val));
}

static float_type ufixed_to_float(UINT_TYPE ufixed_val) {
    return fixed_to_float(twos_complement_to_int(ufixed_val));
}


};

// Specialization for the case where float_type and UINT_TYPE are both float

template <int fractional>
struct FloatFixedConverter<float, float, float, fractional> {
    static float fixed_to_float(float val) {
        return val;
    }

    static float float_to_fixed(float val) {
        return val;
    }

    static float int_to_twos_complement(float val) {
        return val;
    }

    static float twos_complement_to_int(float val) {
        return val;
    }

    static float float_to_ufixed(float val) {
        return val;
    }

    static float ufixed_to_float(float val) {
        return val;
    }
};

template <typename float_type, typename INT_TYPE, typename UINT_TYPE, int fractional>
float_type fixedToFloat(UINT_TYPE val) {
    return FloatFixedConverter<float_type, INT_TYPE, UINT_TYPE, fractional>::fixedToFloat(val);
}

template <typename float_type, typename INT_TYPE, typename UINT_TYPE, int fractional>
UINT_TYPE floatToFixed(float_type val) {
    return FloatFixedConverter<float_type, INT_TYPE, UINT_TYPE, fractional>::floatToFixed(val);
}

