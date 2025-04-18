#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/Relu.hpp"
#include "../functions/comparisons.hpp"
#include "../functions/max_min.hpp"
#include "../functions/share_conversion.hpp"

#define RESULTTYPE DATATYPE
#define FUNCTION Mixed_Circuits_Tutorial

template <typename Share>
void Mixed_Circuits_Tutorial(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;  // Additive Share to store secret shares in the arithmetic domain
    using S = XOR_Share<DATATYPE, Share>;       // XOR Share to store secret shares in the boolean domain

    const int vectorization_factor = DATTYPE / BITLENGTH;
    //---------------------------------//
    // The AB programming model
    // The Framework switches between arithemtic vectorization for arithemtic operations and Bitslicing for boolean
    // operations. Covnerting between these domains is a low level procedure and we recommend using the high level
    // functions from the functions directory that take care of this conversion.
    const int len = 1000;
    A listA[len];
    A listB[len];
    for (int i = 0; i < len; i++)
    {
        listA[i] = i;
        listB[i] = len - i;
    }

    //---------------------------------//
    // Comparisons
    // Helper functions for comparisons can be imported from the functions directory. The following procedure compares
    // if the values in listA are smaller than the values in listB.
    A result[len];
    for (int i = 0; i < len; i++)
    {
        result[i] =
            listA[i] - listB[i];  // if the result is negative, the value in listA is smaller than the value in listB
    }
    A comparison_result[len];
    pack_additive<0, BITLENGTH>(
        result,
        comparison_result,
        len,
        LTZ<0, BITLENGTH, Share, DATATYPE>);  // The LTZ function returns 1 if the value is negative and 0 otherwise.
                                              // The result is in the arithmetic domain and already handles all
                                              // transfomrations in a communication efficient way.

    pack_additive<0, BITLENGTH>(
        result,
        comparison_result,
        len,
        EQZ<0, BITLENGTH, Share, DATATYPE>);  // we can also compare equality by checking if the result is equal zero.

    //---------------------------------//
    // ReLU

    // The ReLU function is a common activation function in neural networks. It returns the input value if it is
    // positive and zero otherwise. We support probabilistic relus with reduced BITLENGTH. The follwoing setting
    // considers all bits
    const int rm = 0;  // Changing this to n will ignore the n most significant bits during ReLU computation
    const int rk =
        BITLENGTH;  // Changing this to BITLENGTH - n will ignore the n least significant bits during ReLU computation
    pack_additive_inplace<rm, rk>(listA, len, RELU_range_in_place_opt<rm, rk, Share, DATATYPE>);

    //---------------------------------//
    // Max, Min, Argmax, Argmin
    // The following functions compute the maximum, minimum, argmax and argmin of a batch of independent lists. By
    // processing multiple batches in parallel, the communication overhead is reduced.

    A a[len];
    A b[len];
    for (int i = 0; i < len; i++)
    {
        a[i] = A(i);
        b[i] = A(len - i);
    }
    A max_val[2];
    A min_val[2];
    A argmax[len];
    A argmin[len];
    A merged[2 * len];
    A argmax_merged[2 * len];
    A argmin_merged[2 * len];

    // merge two lists two achieve both computations in equal communication rounds
    for (int i = 0; i < len; i++)
    {
        merged[i] = a[i];
        merged[i + len] = b[i];
    }

    max_min_sint<0, BITLENGTH>(merged, len, max_val, 2, true);   // batch size of 2, want max of each independent array
    max_min_sint<0, BITLENGTH>(merged, len, min_val, 2, false);  // batch size of 2, want min of each independent array
    argmax_argmin_sint<0, BITLENGTH>(merged, len, argmax_merged, 2, true);
    argmax_argmin_sint<0, BITLENGTH>(merged, len, argmin_merged, 2, false);

    A max_from_list_a = max_val[0];
    A max_from_list_b = max_val[1];  // results get stored contiguously in the output array
}
