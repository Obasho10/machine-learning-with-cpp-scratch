
#ifndef LAYER_H
#define LAYER_H
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm> // For std::max
#include "activation.h"

class Layer {
public:
    int inputSize;
    int outputSize;
    std::vector<std::vector<double>> weights;
    std::vector<double> outputs;
    std::vector<double> biases;
    std::vector<double> grads;
    std::vector<double> actoutputs;
    Activation activation;

    // Constructor
    Layer(int inputSize, int outputSize, Activation::Type activationType) 
        : inputSize(inputSize), outputSize(outputSize), activation(activationType)
    {
        // Initialize weights matrix with random values
        weights.resize(outputSize, std::vector<double>(inputSize));
        // Initialize the output vector
        outputs.resize(outputSize, 0.0);
        biases.resize(outputSize, 0.0);
        grads.resize(inputSize, 0.0);
        actoutputs.resize(outputSize, 0.0);
        initializeWeights();
    }

    // Copy Constructor
    Layer(const Layer& other) 
        : inputSize(other.inputSize), outputSize(other.outputSize), 
          weights(other.weights), outputs(other.outputs) ,biases(other.biases) ,grads(other.grads),actoutputs(other.actoutputs), activation(other.activation)
    {
        std::cout << "Copy constructor called" << std::endl;
    }

    // Move Constructor
    Layer(Layer&& other) noexcept 
        : inputSize(other.inputSize), outputSize(other.outputSize), 
          weights(std::move(other.weights)), outputs(std::move(other.outputs)) ,biases(std::move(other.biases)),grads(std::move(other.grads)) ,actoutputs(std::move(other.actoutputs)) ,activation(std::move(other.activation))
    {
        std::cout << "Move constructor called" << std::endl;

        // Reset other's members to safe default values
        other.inputSize = 0;
        other.outputSize = 0;
    }

    // Copy Assignment Operator
    Layer& operator=(const Layer& other) {
        if (this != &other) {
            std::cout << "Copy assignment operator called" << std::endl;

            // Perform deep copy of member variables
            inputSize = other.inputSize;
            outputSize = other.outputSize;
            weights = other.weights;
            outputs = other.outputs;
            biases = other.biases;
            grads = other.grads;
            actoutputs = other.actoutputs;
        }
        return *this;
    }

    // Move Assignment Operator
    Layer& operator=(Layer&& other) noexcept {
        if (this != &other) {
            std::cout << "Move assignment operator called" << std::endl;

            // Transfer resources and reset other's members
            inputSize = other.inputSize;
            outputSize = other.outputSize;
            weights = std::move(other.weights);
            outputs = std::move(other.outputs);
            biases = std::move(other.biases);
            grads = std::move(other.grads);
            actoutputs = std::move(other.actoutputs);

            // Reset other's members to safe default values
            other.inputSize = 0;
            other.outputSize = 0;
        }
        return *this;
    }

    // Destructor
    ~Layer() {
        std::cout << "Destructor called" << std::endl;
    }

    // Method to initialize weights with random values
    void initializeWeights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0.0, 1.0);

        for (int i = 0; i < outputSize; ++i) {
            biases[i]=dis(gen);
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] = dis(gen);
            }
        }
    }


    // Method to print the weights (for debugging purposes)
    void printWeights() const {
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                std::cout << weights[i][j] << " ";
            }
            std::cout <<" b "<< biases[i] << " ";
            std::cout << std::endl;
        }
    }

    // Method to print the outputs (for debugging purposes)
    void printOutputs() const {
        for (const auto& output : outputs) {
            std::cout << output << " ";
        }
        std::cout << std::endl;
    }

    void forwardPass(const std::vector<double>& input) {
        // Check that input size matches
        if (input.size() != static_cast<std::vector<double>::size_type>(inputSize)) {
            std::cerr << "Error: Input size does not match layer input size!" << std::endl;
            return;
        }

        // Calculate the output for each neuron
        for (int i = 0; i < outputSize; i++) {
            outputs[i] = biases[i];
            for (int j = 0; j < inputSize; j++) {
                outputs[i] += weights[i][j] * input[j];
            }
        }   
        actoutputs = activation.activate(outputs);      
    }

    // Update weights based on gradients
    std::vector<double> backprop(std::vector<double>& gradients,const std::vector<double>& input, double learningRate) {
        if (gradients.size() != outputs.size()) {
            std::cerr << "Error: Gradient dimensions do not match weight dimensions!" << std::endl;
            return {};
        }
        for (int i = 0; i < outputSize; i++)
        {
            //std::cout<<"i"<<gradients[i]<<std::endl;
            gradients[i]*=activation.derivative(outputs)[i];
        }
        //printWeights();
        for (int i = 0; i < outputSize; ++i) {
            biases[i]-=learningRate*gradients[i];
            //std::cout<<gradients[i]<<std::endl;
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] -= learningRate * gradients[i]*input[j];
            }
        }
        //printWeights();
        
        for (int i = 0; i < inputSize; i++)
        {
            grads[i]=0.0;
            for (int j = 0; j < outputSize; j++)
            {
                //std::cout<<"j"<<gradients[j]<<std::endl;
                grads[i]+=weights[j][i]*gradients[j];
            }
            //std::cout<<grads[i]<<std::endl;
            
        }
        return grads;
    }


    void saveWeights(const std::string& filename) const {
    std::ofstream outFile(filename);
    
    if (outFile.is_open()) {
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                outFile << weights[i][j] << " ";
            }
            outFile << " b " << biases[i] << "\n";
        }
        outFile.close();
        std::cout << "Weights and biases saved to " << filename << std::endl;
    } else {
        std::cerr << "Failed to open the file." << std::endl;
    }
}



    void loadWeights(const std::string& filename) {
    std::ifstream inFile(filename);
    
    if (inFile.is_open()) {
        for (int i = 0; i < outputSize; ++i) {
            std::string line;
            if (std::getline(inFile, line)) {
                std::istringstream iss(line);
                for (int j = 0; j < inputSize; ++j) {
                    iss >> weights[i][j];
                }
                std::string dummy; // to read the "b" character
                iss >> dummy >> biases[i];
            } else {
                std::cerr << "Error: Not enough lines in the file to load all weights and biases." << std::endl;
                break;
            }
        }
        inFile.close();
        std::cout << "Weights and biases loaded from " << filename << std::endl;
    } else {
        std::cerr << "Failed to open the file." << std::endl;
    }
}


};

#endif // LAYER_H