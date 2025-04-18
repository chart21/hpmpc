#include <conv2d_NHWC.cuh>
#include <transform.cuh>
#include <cstdint>
#include <cstdlib>
#include <string>
// constexpr const char* PIGEON_LAYOUT = "NCHW"; // USE_CUDA_GEMM 2
// constexpr const char* PIGEON_LAYOUT = "CHWN"; // USE_CUDA_GEMM 4

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
                    int dilation = 1)
{
    Type* x;
    Type* w;
    Type* y;
    Type* xt;
    Type* wt;
    Type* yt;
    int xSize = inh * inw * din * batchSize;
    int wSize = wh * ww * din * dout;
    int outh = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
    int outw = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;
    int ySize = outh * outw * dout * batchSize;
    cudaMalloc((void**)&x, xSize * sizeof(Type));
    cudaMemcpy(x, X, xSize * sizeof(Type), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&xt, xSize * sizeof(Type));
    if constexpr (PIGEON_LAYOUT == "CHWN")
    {
        chwn_to_nhwc_(xt, x, batchSize, inh, inw, din);
    }
    else if constexpr (PIGEON_LAYOUT == "NCHW")
    {
        nchw_to_nhwc_(xt, x, batchSize, inh, inw, din);
    }
    cudaFree(x);

    cudaMalloc((void**)&w, wSize * sizeof(Type));
    cudaMemcpy(w, W, wSize * sizeof(Type), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&wt, wSize * sizeof(Type));
    if constexpr (PIGEON_LAYOUT != "NHWC")
    {
        nchw_to_nhwc_(wt, w, dout, wh, ww, din);
    }
    cudaFree(w);

    cudaMalloc((void**)&y, ySize * sizeof(Type));

    gpu::conv_fprop<Type>(xt, wt, y, batchSize, inh, inw, din, dout, wh, ww, padding, padding, stride, dilation);

    cudaFree(xt);
    cudaFree(wt);

    cudaMalloc((void**)&yt, ySize * sizeof(Type));
    if constexpr (PIGEON_LAYOUT == "CHWN")
    {
        nhwc_to_chwn_(yt, y, batchSize, outh, outw, dout);
    }
    else if constexpr (PIGEON_LAYOUT == "NCHW")
    {
        nhwc_to_nchw_(yt, y, batchSize, outh, outw, dout);
    }

    cudaMemcpy(Y, yt, ySize * sizeof(Type), cudaMemcpyDeviceToHost);

    cudaFree(yt);
}

// UINT8 and UINT16 are not supported by all architectures
// template void conv2d_cutlass<uint8_t>(const uint8_t* X, const uint8_t* W, uint8_t* Y, int batchSize, int inh, int
// inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation);
template void conv2d_cutlass<uint16_t>(const uint16_t* X,
                                       const uint16_t* W,
                                       uint16_t* Y,
                                       int batchSize,
                                       int inh,
                                       int inw,
                                       int din,
                                       int dout,
                                       int wh,
                                       int ww,
                                       int padding,
                                       int stride,
                                       int dilation);  // INT8 and INT16 are not supported by all architectures
template void conv2d_cutlass<uint32_t>(const uint32_t* X,
                                       const uint32_t* W,
                                       uint32_t* Y,
                                       int batchSize,
                                       int inh,
                                       int inw,
                                       int din,
                                       int dout,
                                       int wh,
                                       int ww,
                                       int padding,
                                       int stride,
                                       int dilation);
template void conv2d_cutlass<uint64_t>(const uint64_t* X,
                                       const uint64_t* W,
                                       uint64_t* Y,
                                       int batchSize,
                                       int inh,
                                       int inw,
                                       int din,
                                       int dout,
                                       int wh,
                                       int ww,
                                       int padding,
                                       int stride,
                                       int dilation);
