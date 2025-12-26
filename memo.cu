#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <opencv4/opencv/2/opencv.hpp>

#define LENGTH 1024

#define NUM_DATA 10

__global__ void kernel(int *dA, int *dB, int *dC)
{
    dA[threadIdx.x] = threadIdx.x;
}

void checkDeviceMemory(void)
{
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("Device memory - free: %zu bytes, total: %zu bytes\n", free, total);
}

int main(void)
{
    int ptrHostA[LENGTH];
    int ptrHostB[LENGTH];
    int ptrHostC[LENGTH];

    for (int i = 0; i < LENGTH; i++)
    {
        ptrHostA[i] = rand() % 256;
        ptrHostB[i] = rand() % 256;
        ptrHostC[i] = rand() % 256;
    }
    
    int* ptrDeviceA;
    int* ptrDeviceB;
    int* ptrDeviceC;

    unsigned int arrSize = sizeof(int) * LENGTH;

    cudaMalloc((void**)&ptrDeviceA, sizeof(int) * LENGTH); cudaMemset(ptrDeviceA, 0, arrSize);
    cudaMalloc((void**)&ptrDeviceB, sizeof(int) * LENGTH); cudaMemset(ptrDeviceB, 0, arrSize);
    cudaMalloc((void**)&ptrDeviceC, sizeof(int) * LENGTH); cudaMemset(ptrDeviceC, 0, arrSize);

    cudaMemcpy(ptrDeviceA, ptrHostA, arrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(ptrDeviceB, ptrHostB, arrSize, cudaMemcpyHostToDevice);
    cudaMemcpy(ptrDeviceC, ptrHostC, arrSize, cudaMemcpyHostToDevice);

    kernel<<<1, 1>>>(dDataPtr);

    cudaDeviceSynchronize();

    cudaMemcpy(ptrDeviceA, ptrHostA, arrSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ptrDeviceB, ptrHostB, arrSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(ptrDeviceC, ptrHostC, arrSize, cudaMemcpyDeviceToHost);

    cudaFree(ptrDeviceA);
    cudaFree(ptrDeviceB);
    cudaFree(ptrDeviceC);

    return 0;
}