#include "mat_mul.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void MatMulKernel(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row >= m || col >= n)
        return;

    DATA_TYPE val = 0;
    for (int i = 0; i < k; i++)
        val += KERNEL_MUL(matA[ID2INDEX(row, i, k)], matB[ID2INDEX(i, col, n)]);

    matC[ID2INDEX(row, col, n)] = val;
}

void launchMatMulKernel(DATA_TYPE* dA, DATA_TYPE* dB, DATA_TYPE* dC, int m, int n, int k)
{
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    MatMulKernel<<<gridDim, blockDim>>>(dA, dB, dC, m, n, k);
    cudaDeviceSynchronize(); // 커널 완료 대기

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(err) << std::endl;
}
