#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloFromGPU(void) 
{
    printf("Hello from GPU!\n");
}

int main(void) 
{
    // Launch kernel with 1 block and 1 thread
    helloFromGPU<<<1, 10>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return 0;
}

