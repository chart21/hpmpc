#pragma once
#include "cutlass/cutlass.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/tensor_ref.h"
#include <cutlass/util/device_nchw_to_nhwc.h>
#include <cutlass/util/device_nhwc_to_nchw.h>

// macro for transpose only
#define TILE_DIM 64
#define BLOCK_ROWS 16

template <typename T>
void nchw_to_nhwc_(T* output, const T* input, const int n, const int h, const int w, const int c)
{

    dim3 grid((h * w + 31) / 32, (c + 31) / 32, n);
    dim3 block(32, 8);
    cutlass::nchw_to_nhwc_kernel<<<grid, block, 0, 0>>>(output, input, n, h, w, c);
}

template <typename T>
__global__ void transposeGPUcoalescing(T* matTran, const T* matIn, const int nx, const int ny)
{
    __shared__ T tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;  // <- threadIdx.y only between 0 and 7
    // Load matrix into tile
    // Every Thread loads in this case 4 elements into tile.
    int i;
    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (x < nx && (y + i) < ny)
        {
            tile[threadIdx.y + i][threadIdx.x] = matIn[(y + i) * nx + x];
        }
    }
    __syncthreads();
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (x < ny && (y + i) < nx)
        {
            matTran[(y + i) * ny + x] = tile[threadIdx.x][threadIdx.y + i];  // <- multiply by m, non-squared!
        }
    }
}

template <typename T>
void transpose(T* output, const T* input, const int cols, const int rows)
{

    int x = cols;
    int y = rows;

    if (cols % TILE_DIM)
        x += (TILE_DIM - (cols % TILE_DIM));
    if (rows % TILE_DIM)
        y += (TILE_DIM - (rows % TILE_DIM));

    dim3 dimGrid(x / TILE_DIM, y / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    transposeGPUcoalescing<<<dimGrid, dimBlock>>>(output, input, cols, rows);
}

template <typename T>
void chwn_to_nhwc_(T* output, const T* input, const int n, const int h, const int w, const int c)
{

    T* buf;
    int size = n * c * h * w;
    cudaMalloc((void**)&buf, sizeof(T) * size);
    int rows = c * h * w;
    int cols = n;
    // chwn -> nchw
    transpose(buf, input, cols, rows);
    // nchw -> nhwc
    nchw_to_nhwc_(output, buf, n, h, w, c);
    cudaFree(buf);
}

// old approach
// template <typename T>
// void chwn_to_nhwc_(
//    T *output,
//    const T *input,
//    const int n,
//    const int h,
//    const int w,
//    const int c) {
//
//    int newN = 1;
//    int newC = c*h*w;
//    int newH = 1;
//    int newW = n;
//
//    T* buf;
//    int size = n*c*h*w;
//    cudaMalloc((void**) &buf, sizeof(T)*size);
//
//    nchw_to_nhwc_(buf, input, newN, newH, newW, newC);
//    nchw_to_nhwc_(output, buf, n, h, w, c);
//
//    cudaFree(buf);
//}

template <typename T>
void nhwc_to_nchw_(T* output, const T* input, const int n, const int h, const int w, const int c)
{

    dim3 grid((c + 31) / 32, (h * w + 31) / 32, n);
    dim3 block(32, 8);
    cutlass::nhwc_to_nchw_kernel<<<grid, block, 0, 0>>>(output, input, n, h, w, c);
}

template <typename T>
void nhwc_to_chwn_(T* output, const T* input, const int n, const int h, const int w, const int c)
{

    int newN = c;
    int newC = n;

    chwn_to_nhwc_(output, input, newN, h, w, newC);
}
