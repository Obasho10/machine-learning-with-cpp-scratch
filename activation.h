#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>  // For std::invalid_argument


class Activation {
public:
    // Enumeration for different activation types
    enum class Type {
        Sigmoid,
        ReLU,
        SoftMax,
        Default
    };

    // Constructor to set the activation type
    Activation(Type type) : activationType(type) {}

    // Function to apply the chosen activation function
    std::vector<double> activate(const std::vector<double>& input) const {
        switch (activationType) {
            case Type::Sigmoid:
                return sigmoid(input);
            case Type::ReLU:
                return relu(input);
            case Type::SoftMax:
                return softmax(input);
            case Type::Default:
            default:
                return input;
        }
    }

    // Function to apply the derivative of the chosen activation function
    std::vector<double> derivative(const std::vector<double>& input) const {
        switch (activationType) {
            case Type::Sigmoid:
                return sigmoidDerivative(input);
            case Type::ReLU:
                return reluDerivative(input);
            case Type::SoftMax:
                return softmax_derivative(input);
            case Type::Default:
            default:{
            std::vector<double> ones(input.size(), 1.0);
            return ones;
        }
        }
    }

private:
    Type activationType;
    // Function to compute the softmax of a vector
    std::vector<double> softmax(const std::vector<double>& input) const{
        std::vector<double> output(input.size());

        double max_val = *std::max_element(input.begin(), input.end());
        std::vector<double> exp_values(input.size());
        double sum_exp = 0.0;

        for (size_t i = 0; i < input.size(); ++i) {
            exp_values[i] = std::exp(input[i] - max_val);
            sum_exp += exp_values[i];
        }

        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = exp_values[i] / sum_exp;
        }

        return output;
    }

    // Function to compute the derivative of the softmax of a vector
    std::vector<double> softmax_derivative(const std::vector<double>& input) const{
        std::vector<double> softmax_values = softmax(input);
        size_t n = softmax_values.size();

        std::vector<double> derivative(input.size());

        for (size_t i = 0; i < n; ++i) {
            derivative[i]=0.0;
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    derivative[i] += softmax_values[i] * (1 - softmax_values[i]);
                } else {
                    derivative[i] -= softmax_values[i] * softmax_values[j];
                }
            }
        }
        

        return derivative;
    }

    // Sigmoid function
    std::vector<double> sigmoid(const std::vector<double>& input) const {
        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = 1.0 / (1.0 + std::exp(-input[i]));
        }
        return output;
    }

    // Derivative of the sigmoid function
    std::vector<double> sigmoidDerivative(const std::vector<double>& input) const {
        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            double sigmoidValue = 1.0 / (1.0 + std::exp(-input[i]));
            output[i] = sigmoidValue * (1.0 - sigmoidValue);
            
        }
        return output;
    }

    // ReLU function
    std::vector<double> relu(const std::vector<double>& input) const {
        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::min(1.0,std::max(0.0, input[i]));
        }
        return output;
    }

    // Derivative of the ReLU function
    std::vector<double> reluDerivative(const std::vector<double>& input) const {
        std::vector<double> output(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = ((input[i] > 0)&&input[i]<1) ? 1.0 : 0.0;
        }
        return output;
    }
};

#endif // ACTIVATION_H
