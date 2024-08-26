#ifndef MODEL_H
#define MODEL_H

#include <iostream>
#include <vector>
#include <memory>
#include "layer.h"
# include "loss.h"
#include "activation.h"
#include <filesystem>



class Model {
public:
    LossType lossType;
    std::vector<double> inputs;
    std::vector<std::unique_ptr<Layer>> layers;
    // Method to add a layer
    void addLayer(int inputSize, int outputSize,Activation::Type activation) {
        layers.emplace_back(std::make_unique<Layer>(inputSize, outputSize,activation));
    }

    // Forward pass through the network
    void forwardPass() {
        std::vector<double> currentInput = inputs;
        for (auto& layer : layers) {
            layer->forwardPass(currentInput);
            currentInput = layer->outputs;  // Output of current layer is input to the next layer
        }
    }

    // Backpropagate the loss and update weights
    Model(LossType lossType) : lossType(lossType) {}
    double backpropagate(const std::vector<double>& trueValues, double learningRate) {
    // Calculate the gradient vector for the output layer using the MSE and its gradient
    std::vector<double> output = layers.back()->outputs;  // Assuming there's a method to get the output of the last layer
    auto [mse, lossGradient] = calculateLossAndGrad(trueValues, output);
    //std::cout<<lossGradient[0]<<std::endl;
    // Start backpropagating from the last layer to the first hidden layer
    std::vector<double> inputGradients = lossGradient;
    
    for (int l = layers.size() - 1; l > 0; --l) {
        // Get the input to the current layer
        std::vector<double> input = layers[l-1]->actoutputs;  // Assuming there's a method to get the input to the current layer
        //layers[l]->printWeights();
        // Backpropagate through the current layer
        inputGradients = layers[l]->backprop(inputGradients, input, learningRate);
        //std::cout<<inputGradients[0]<<std::endl;
    }
    layers[0]->backprop(inputGradients,inputs,learningRate);
    //layers[0]->printWeights();
    return mse;

}
    double trainStep(const std::vector<double>& input, const std::vector<double>& trueValues, double learningRate) {
    // Set the inputs for the model
    inputs = input;
    //std::cout<<"in"<<inputs[0]<<"\t"<<inputs[1]<<std::endl;
    // Perform the forward pass
    forwardPass();

    // Perform the backpropagation to update the weights
    double mse=backpropagate(trueValues, learningRate);
    return mse;
}
    std::pair<double, std::vector<double>> calculateLossAndGrad(const std::vector<double>& trueValues, const std::vector<double>& output) {
        Loss lossCalculator;
        if (lossType == LossType::MSE) {
            return lossCalculator.calculateMSEandGrad(trueValues, output);
        } else if (lossType == LossType::BCE) {
            return lossCalculator.calculateBCEandGrad(trueValues, output);
        } else {
            throw std::invalid_argument("Unknown Loss Type");
        }
    }



    void saveModelWeights(const std::string& folderName) const {
    // Create the directory if it doesn't exist
    std::filesystem::create_directories(folderName);

    // Iterate over each layer and save its weights to a separate file
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string filename = folderName + "/layer_" + std::to_string(i) + ".txt";
        layers[i]->saveWeights(filename);
    }

    std::cout << "Model weights and biases saved in folder: " << folderName << std::endl;
}

    void loadModelWeightsFromFolder(const std::string& folderName) {
    // Iterate over each layer and load its weights from the respective file
    for (size_t i = 0; i < layers.size(); ++i) {
        std::string filename = folderName + "/layer_" + std::to_string(i) + ".txt";
        layers[i]->loadWeights(filename);
    }

    std::cout << "Model weights and biases loaded from folder: " << folderName << std::endl;
}

};

#endif // MODEL_H