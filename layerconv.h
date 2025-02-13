#pragma once

#include <vector>

class layers {
private:
    int outputSize;
    int inputSize;
    int padding;
    int inputLayers;
    int outputLayers;
    int kernal;
    dim3 threadsPerBlock;
    dim3 numBlocksinput;
    dim3 numBlocksoutput;
    float* inputMatricesGPU[2];
    float* outputMatricesGPU[2];
    float* outputWeightMatricesGPU;
    float* inputWeightMatricesGPU;

public:
    layers(int outSize, int inSize, int pad, int inLayers, int outLayers,int Kernal);
    ~layers();
    
    void copyInputToGPU(const std::vector<std::vector<std::vector<float>>>& inputData, int index);
    void copyOutputToGPU(const std::vector<std::vector<std::vector<float>>>& outputData, int index);
    void copyInputToCPU(std::vector<std::vector<std::vector<float>>>& inputData, int index);
    void copyOutputToCPU(std::vector<std::vector<std::vector<float>>>& outputData, int index);
    void copyInputWeightsToGPU(const std::vector<std::vector<std::vector<float>>>& inputWeights);
    void copyOutputWeightsToGPU(const std::vector<std::vector<std::vector<float>>>& outputWeights);
    void copyInputWeightsToCPU(std::vector<std::vector<std::vector<float>>>& inputWeights);
    void copyOutputWeightsToCPU(std::vector<std::vector<std::vector<float>>>& outputWeights);
    void calculate_weights(int kernel_size,int inlayers, float sigma, float *kernel);
    void InputConv(int padding);
};
