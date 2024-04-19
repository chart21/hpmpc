#pragma once
#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include "conv_cutlass_int.h"

void test_conv() {

// sample 3X3 image
uint32_t X[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
uint32_t W[4] = {1, 1, 1, 1}; // 2x2 filter
uint32_t Y[4]; // Output

// Initialize Y to zero
memset(Y, 0, sizeof(Y));

    // Call convolution function
conv2d_cutlass(X,W,Y,1,3,3,1,1,2,2,0,1,1);
/* conv2d_cutlass(const Type *X, const Type *W, Type *Y, int batchSize, int inh, int inw, int din, int dout, int wh, int ww, int padding, int stride); */

// Print output
for (int i = 0; i < 4; i++) 
    std::cout << Y[i] << " ";

}

