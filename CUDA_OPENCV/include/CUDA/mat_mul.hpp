#pragma once
#include "cuda_runtime.h"

#define BLOCK_SIZE 16
#define DATA_TYPE int

// Kernel 매크로
#define KERNEL_MUL(_a,_b) ((_a)*(_b))

// ID/Index 매크로
#define ID2INDEX(_row,_col,_width) (((_row)*(_width))+(_col))

// CUDA kernel 선언
__global__ void MatMulKernel(DATA_TYPE* matA, DATA_TYPE* matB, DATA_TYPE* matC, int m, int n, int k);

// GPU 실행 wrapper
void launchMatMulKernel(DATA_TYPE* dA, DATA_TYPE* dB, DATA_TYPE* dC, int m, int n, int k);
