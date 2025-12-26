#include <cuda_runtime.h>
#include "CUDA_OpenCV/kernel.hpp"

__global__ void inversionKernel(unsigned char* d, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    d[idx] = 255 - d[idx];
}

__global__ void matMul_kernel(unsigned char* d, int width, int height)
{
    
}

void launchKernel(unsigned char* d, int width, int height)
{
    dim3 block(32, 32);
    dim3 grid(
        ceil((float)width  / block.x),
        ceil((float)height / block.y)
    );

    inversionKernel<<<grid, block>>>(d, width, height);
}