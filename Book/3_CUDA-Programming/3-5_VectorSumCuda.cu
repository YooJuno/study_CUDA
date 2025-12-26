#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The Size of the vector
#define NUM_DATA 1024

// Simple vector sum kernel
__global__ void vecAdd(int* _a, int* _b, int* _c) 
{
    int tID = threadIdx.x;
    _c[tID] = _a[tID] + _b[tID];
}

void checkDeviceMemory(void)
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Device memory - free: %zu bytes, total: %zu bytes\n", free, total);
}

int main() 
{
    int *hostA, *hostB, *hostC, *hostHC;
    int *deviceA, *deviceB, *deviceC;

    cudaError_t errorCode;

    int memSize = sizeof(int) * NUM_DATA;
    printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

    // Memory allocation on the host-side
    hostA = new int[NUM_DATA]; memset(hostA, 0, memSize);
    hostB = new int[NUM_DATA]; memset(hostB, 0, memSize);
    hostC = new int[NUM_DATA]; memset(hostC, 0, memSize);
    hostHC = new int[NUM_DATA]; memset(hostHC, 0, memSize);

    // Data generation
    for (int i = 0; i < NUM_DATA; i++) 
    {
        hostA[i] = rand() % 10;
        hostB[i] = rand() % 10;
    }

    // Vector sum on host (for performance comparison)
    for (int i = 0; i < NUM_DATA; i++) 
    {
        hostHC[i] = hostA[i] + hostB[i];
    }

    
    
    // Memory allocation on the device-side
    errorCode = cudaMalloc(&deviceA, memSize); errorCode = cudaMemset(deviceA, 0, memSize);
    errorCode = cudaMalloc(&deviceB, memSize); errorCode = cudaMemset(deviceB, 0, memSize);
    errorCode = cudaMalloc(&deviceC, memSize); errorCode = cudaMemset(deviceC, 0, memSize);  

    

    // Data copy : Host -> Device
    // (void *dst, const void *src, size_t count, cudaMemcpyKind kind)
    errorCode = cudaMemcpy(deviceA, hostA, memSize, cudaMemcpyHostToDevice);
    errorCode = cudaMemcpy(deviceB, hostB, memSize, cudaMemcpyHostToDevice);

    // Kernel call
    vecAdd<<<1, NUM_DATA>>>(deviceA, deviceB, deviceC);
    
    // Copy result : Device -> Host
    errorCode = cudaMemcpy(hostC, deviceC, memSize, cudaMemcpyDeviceToHost);

    // Release device memory
    errorCode = cudaFree(deviceA);
    errorCode = cudaFree(deviceB);
    errorCode = cudaFree(deviceC);

    // Check results
    bool result = true;
    for(int i = 0; i < NUM_DATA; i++) 
    {
        if (hostHC[i] != hostC[i]) 
        {
            result = false;
            printf("[%d] The result is not matched! (%d, %d)\n", i, hostHC[i], hostC[i]);
            break;
        }
    }   

    if(result) 
    {
        printf("GPU works well!\n");
    }

    // Release host memory
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    delete[] hostHC;
    
    return 0;
}