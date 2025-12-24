#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

void checkDeviceMemory(void)
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Device memory - free: %zu bytes, total: %zu bytes\n", free, total);
}

int main(void)
{
    int* dDataPtr;
    cudaError_t errorCode;

    checkDeviceMemory();
    errorCode = cudaMalloc(&dDataPtr, sizeof(int) * 1024 * 1024);
    printf("After cudaMalloc, error code: %d\n", errorCode);
    checkDeviceMemory();

    errorCode = cudaMemset(dDataPtr, 0, sizeof(int) * 1024 * 1024);
    printf("After cudaMemset, error code: %d\n", errorCode);

    errorCode = cudaFree(dDataPtr);
    printf("After cudaFree, error code: %d\n", errorCode);
    checkDeviceMemory();
}