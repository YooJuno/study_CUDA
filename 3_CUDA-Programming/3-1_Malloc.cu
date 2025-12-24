#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    int* dDataPtr;
    cudaMalloc(&dDataPtr, sizeof(int) * 32);

    return 0;
}