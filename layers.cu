#include "layerconv.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda.h>
#include<iostream>
#include <cuda_runtime.h>
#define pi 3.14159265359
cudaError_t cudaStatus;


__global__ void conv_img_gpu(float *img, float *kernel, float *imgf, int kernel_size)
{
  int N=gridDim.x*blockDim.x;  
  //local ID of each thread (withing block) 
  int ix = threadIdx.x+blockIdx.x*blockDim.x;    
  //each block is assigned to a row of an image, iy index of y value                  
  int iy = threadIdx.y + blockIdx.y*blockDim.y;  
  //each thread is assigned to a pixel of a row, ix index of x value
  int iz = threadIdx.z ; 
  //idx global index (all blocks) of the image pixel 
  int idx = iz*N*N+iy*N +ix;                        
 //total number of kernel elements
  int K2 = kernel_size*kernel_size;  
  //center of kernel in both dimensions          
  int center = (kernel_size -1)/2;		 
  //Auxiliary variables
  int ii, jj;
  float sum = 0.0;
 /*
 Define a vector (float) sdata[] that will be hosted in shared memory,
 *extern* dynamic allocation of shared memory: kernel<<<blocks,threads,memory size to be allocated in shared memory>>>
*/  
/*
  Convlution of image with the kernel
  Each thread computes the resulting pixel value 
  from the convolution of the original image with the kernel;
  number of computations per thread = size_kernel^2
  The result is stored to imgf
  */
  if (idx<N*N*8)
  {
    for (int ki = 0; ki<kernel_size; ki++)
      for (int kj = 0; kj<kernel_size; kj++)
        for (int kk = 0;kk<8;kk++)
        {
            ii = (ix-center)+ki;
            jj = (iy-center)+kj;
            if(ii<0 || jj<0 || ii>N || jj>N)
            {
                sum+=0;
                continue;
            }
            sum+=img[kk*N*N+jj*N+ii]*kernel[kk*K2+kj*kernel_size + ki];
        }
    imgf[idx] = sum;
  }
}

layers::layers(int outSize, int inSize, int pad, int inLayers, int outLayers,int ker) 
    : outputSize(outSize), inputSize(inSize), padding(pad), inputLayers(inLayers), outputLayers(outLayers),kernal(ker) {
    for (int i = 0; i < 2; ++i) {
        cudaMalloc(&inputMatricesGPU[i], inputLayers * inputSize * inputSize * sizeof(float));
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) 
        {
            std::cout<<cudaGetErrorString(cudaStatus);
        }
        else
            std::cout<<"sucess copy"<<std::endl;
        cudaMalloc(&outputMatricesGPU[i], outputLayers * outputSize * outputSize * sizeof(float));
    }
    cudaMalloc(&inputWeightMatricesGPU, inputLayers * inputLayers * kernal*kernal * sizeof(float));
    cudaMalloc(&outputWeightMatricesGPU, outputLayers * inputLayers * kernal*kernal * sizeof(float));
    cudaStatus = cudaGetLastError();
    std::cout<<'0';
    if (cudaStatus != cudaSuccess) 
    {
        std::cout<<cudaGetErrorString(cudaStatus);
    }
    else
        std::cout<<"sucess copy"<<std::endl;
    calculate_weights(kernal,inputLayers, 1,inputWeightMatricesGPU);
    std::cout<<'0';
    calculate_weights(kernal,outputLayers,1, outputWeightMatricesGPU);
    

}

layers::~layers() {
    for (int i = 0; i < 2; ++i) {
        cudaFree(inputMatricesGPU[i]);
        cudaFree(outputMatricesGPU[i]);
    }
    cudaFree(outputWeightMatricesGPU);
    cudaFree(inputWeightMatricesGPU);
}

void layers::copyInputToGPU(const std::vector<std::vector<std::vector<float>>>& inputData, int index) {
    cudaMemcpy(inputMatricesGPU[index], inputData.data(), inputLayers * inputSize * inputSize * sizeof(float), cudaMemcpyHostToDevice);
}

void layers::copyOutputToGPU(const std::vector<std::vector<std::vector<float>>>& outputData, int index) {
    cudaMemcpy(outputMatricesGPU[index], outputData.data(), outputLayers * outputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);
}

void layers::copyInputToCPU(std::vector<std::vector<std::vector<float>>>& inputData, int index) {
    cudaMemcpy(inputData.data(), inputMatricesGPU[index], inputLayers * inputSize * inputSize * sizeof(float), cudaMemcpyDeviceToHost);
}

void layers::copyOutputToCPU(std::vector<std::vector<std::vector<float>>>& outputData, int index) {
    cudaMemcpy(outputData.data(), outputMatricesGPU[index], outputLayers * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
}

void layers::copyInputWeightsToGPU(const std::vector<std::vector<std::vector<float>>>& inputWeights) {
    cudaMemcpy(inputWeightMatricesGPU, inputWeights.data(), inputLayers * inputLayers * kernal*kernal * sizeof(float), cudaMemcpyHostToDevice);
}

void layers::copyOutputWeightsToGPU(const std::vector<std::vector<std::vector<float>>>& outputWeights) {
    cudaMemcpy(outputWeightMatricesGPU, outputWeights.data(), outputLayers * outputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);
}

void layers::copyInputWeightsToCPU(std::vector<std::vector<std::vector<float>>>& inputWeights) {
    cudaMemcpy(inputWeights.data(), inputMatricesGPU,inputLayers * inputLayers * kernal*kernal * sizeof(float), cudaMemcpyDeviceToHost);
}

void layers::copyOutputWeightsToCPU(std::vector<std::vector<std::vector<float>>>& outputWeights) {
    cudaMemcpy(outputWeights.data(), outputMatricesGPU, outputLayers * outputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
}

void layers::InputConv(int padding)
{
    dim3 blockSize;
    if(inputSize<8)
        dim3 blockSize(inputSize,inputSize,8);
    else
        dim3 blockSize(8,8,8);
    std::cout<<blockSize.x;
    dim3 gridSize(inputSize/blockSize.x,inputSize/blockSize.y,1);
    conv_img_gpu<<<gridSize, blockSize>>>(inputMatricesGPU[0],inputWeightMatricesGPU, inputMatricesGPU[1], kernal);
    cudaDeviceSynchronize(); // Wait for kernel to finish
}

void layers::calculate_weights(int kernel_size, int inlayers, float sigma, float *kernel)
{
    int Nk3 = kernel_size * kernel_size * inputLayers;
    float center = (kernel_size - 1) / 2.0f;
    for (int i = 0; i < Nk3; i++)
    {
        int z_idx = i / (kernel_size * kernel_size);
        int rem = i % (kernel_size * kernel_size);
        int y_idx = rem / kernel_size;
        int x_idx = rem % kernel_size;

        float x = x_idx - center;
        float y = y_idx - center;
        float z = z_idx - center;

        kernel[i] = -(1.0f / (pi * pow(sigma, 6))) * (1.0f - 0.5f * ((x * x + y * y + z * z) / (sigma * sigma))) * exp(-0.5f * ((x * x + y * y + z * z) / (sigma * sigma)));
    }
}
