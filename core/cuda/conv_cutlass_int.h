#pragma once
#include <cstdint>

template <typename Type>
void conv2d_cutlass(const Type* X,
                    const Type* W,
                    Type* Y,
                    int batchSize,
                    int inh,
                    int inw,
                    int din,
                    int dout,
                    int wh,
                    int ww,
                    int padding,
                    int stride,
                    int dilation = 1);
