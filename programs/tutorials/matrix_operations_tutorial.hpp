#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/GEMM.hpp"

#define RESULTTYPE DATATYPE
#define FUNCTION Matrix_Operations_Tutorial

template <typename Share>
void Matrix_Operations_Tutorial(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    using S = XOR_Share<DATATYPE, Share>;
    const int vectorization_factor = DATTYPE / BITLENGTH;
    //---------------------------------//
    // The ABG programming model
    // The Framework switches between arithemtic vectorization for arithemtic operations and Bitslicing for boolean
    // operations. For matrix operations we also provide support for GPU acceleration.This feature is optional and can
    // be enabled by setting USE_CUDA_GEMM to > 0.

    const int m = 3;
    const int k = 2;
    const int n = 1;
    float x[m * k] = {3.2, 5.4, 7.6, 11.8, 2.1, 4.3};
    float w[k * n] = {2.1, 4.3};
    float y[m * n] = {0};
    auto X = new A[m * k];
    for (int i = 0; i < m * k; i++)
    {
        X[i] = A(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(x[i]));
    }
    auto W = new A[k * n];
    for (int i = 0; i < k * n; i++)
    {
        W[i] = A(FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL>::float_to_ufixed(w[i]));
    }

    auto Y = new A[m * n]{A(0)};

    Share::communicate();

    prepare_GEMM(X, W, Y, m, n, k, false);  // automatically uses CPU or GPU depending on the configuration
    Share::communicate();
    complete_GEMM(Y, m * n);
}
