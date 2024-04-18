#include <cstdlib>
#include <conv2d.cuh>
#include <cstdint>

template <typename Type>
void conv2d_cutlass(const Type* X, const Type* W, Type* Y, int batchSize, int inh, int inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation = 1) {
    Type* x;
    Type* w;
    Type* y;
    
    int xSize = inh * inw * din * batchSize;
    int wSize = wh * ww * din * dout;
    int ySize = inh * inw * dout * batchSize;

    cudaMalloc((void **)&x, xSize * sizeof(Type));
    cudaMalloc((void **)&w, wSize * sizeof(Type));
    cudaMalloc((void **)&y, ySize * sizeof(Type));

    cudaMemcpy(x, X, xSize * sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(w, W, wSize * sizeof(Type), cudaMemcpyHostToDevice);

    gpu::conv_fprop<Type>(
        x, w, y,
        batchSize, inh, inw, din, dout,
        wh, ww, padding, padding, stride, dilation
    );

    cudaMemcpy(Y, y, ySize * sizeof(Type), cudaMemcpyDeviceToHost);
}

template void conv2d_cutlass<uint8_t>(const uint8_t* X, const uint8_t* W, uint8_t* Y, int batchSize, int inh, int inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation);
template void conv2d_cutlass<uint16_t>(const uint16_t* X, const uint16_t* W, uint16_t* Y, int batchSize, int inh, int inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation);
template void conv2d_cutlass<uint32_t>(const uint32_t* X, const uint32_t* W, uint32_t* Y, int batchSize, int inh, int inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation);
template void conv2d_cutlass<uint64_t>(const uint64_t* X, const uint64_t* W, uint64_t* Y, int batchSize, int inh, int inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation);
