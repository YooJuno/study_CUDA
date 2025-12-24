#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloFromGPU(void) 
{
    printf("Hello from GPU! Thread (%d, %d, %d) in block (%d, %d, %d)\n", 
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z);
}

int main(void) 
{
    printf("=== 1D Grid - 1D Block ===\n");
    dim3 blockSize1D(2);
    dim3 gridSize1D(2);
    helloFromGPU<<<gridSize1D, blockSize1D>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    printf("\n=== 2D Grid - 1D Block ===\n");
    dim3 blockSize2D1D(2);  // 4 threads per block
    dim3 gridSize2D1D(2, 2);      // 2x2 grid
    helloFromGPU<<<gridSize2D1D, blockSize2D1D>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    printf("\n=== 2D Grid - 2D Block ===\n");
    dim3 blockSize2D(2, 2);    // 2x2 threads per block
    dim3 gridSize2D(2, 2);        // 2x2 grid
    helloFromGPU<<<gridSize2D, blockSize2D>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}

