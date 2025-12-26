#include <cuda_runtime.h>
#include "CUDA_OpenCV/kernel.hpp"

__global__ void myKernel(unsigned char* d, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;
    d[idx] = 255 - d[idx];
}

void launchKernel(unsigned char* d, int width, int height)
{
    dim3 block(32, 32);
    dim3 grid(
        (unsigned int)ceil((float)width  / block.x),
        (unsigned int)ceil((float)height / block.y)
    );

    myKernel<<<grid, block>>>(d, width, height);
}