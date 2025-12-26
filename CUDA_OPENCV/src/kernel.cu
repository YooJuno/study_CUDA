#include <cuda_runtime.h>
#include "cuda_opencv/kernel.hpp"

__global__ void myKernel(unsigned char* d, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) d[idx] = 255 - d[idx];
}

void launchKernel(unsigned char* d, int size)
{
    myKernel<<<(size + 255) / 256, 256>>>(d, size);
}
