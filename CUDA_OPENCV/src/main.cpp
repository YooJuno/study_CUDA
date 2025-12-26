#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "cuda_opencv/kernel.hpp"
#include "DS_Timer/DS_timer.h"

using namespace std;
using namespace cv;

int main()
{
    string pathImgLenna = "/home/juno/Workspace/study_CUDA/CUDA_OPENCV/data/Lenna.png";
    Mat ImgHostA = imread(pathImgLenna, IMREAD_GRAYSCALE);

    if (ImgHostA.empty()) return -1;

    unsigned char* ptrDeviceA;
    int size = ImgHostA.total();

    // Create a copy for CPU processing
    Mat ImgHostCPU = ImgHostA.clone();

    // Initialize timer
    DS_timer timer(5);
    timer.setTimerName(0, "Total CUDA");
    timer.setTimerName(1, "cudaMemcpy H2D");
    timer.setTimerName(2, "Kernel Launch");
    timer.setTimerName(3, "cudaMemcpy D2H");
    timer.setTimerName(4, "CPU Invert");

    timer.timerOn();

    cudaMalloc(&ptrDeviceA, size);

    timer.onTimer(4);
    for (int i = 0; i < size; i++) 
    {
        ImgHostCPU.data[i] = 255 - ImgHostCPU.data[i];
    }
    timer.offTimer(4);

    timer.onTimer(0);
    timer.onTimer(1);
    cudaMemcpy(ptrDeviceA, ImgHostA.data, size, cudaMemcpyHostToDevice);
    timer.offTimer(1);

    timer.onTimer(2);
    launchKernel(ptrDeviceA, size);
    timer.offTimer(2);

    timer.onTimer(3);
    cudaMemcpy(ImgHostA.data, ptrDeviceA, size, cudaMemcpyDeviceToHost);
    timer.offTimer(3);
    timer.offTimer(0);

    cudaFree(ptrDeviceA);

    timer.printTimer();

    imshow("CPU Result", ImgHostCPU);
    imshow("CUDA Result", ImgHostA);
    waitKey(0);

    return 0;
}
