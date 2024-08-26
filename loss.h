

#ifndef LOSS_H
#define LOSS_H

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>


enum class LossType {
    MSE,
    BCE
};


class Loss {
public:
    // Function to calculate MSE and its gradient
    std::pair<double, std::vector<double>> calculateMSEandGrad(const std::vector<double>& trueValue, const std::vector<double>& predicted) {
        if (trueValue.size() != predicted.size()) {
            std::cerr << "Error: The size of trueValue and predicted arrays must be the same!" << std::endl;
            return {0.0, {}};
        }
        
        double mse = 0.0;
        std::vector<double> delMSE(predicted.size(), 0.0);
        
        for (size_t i = 0; i < trueValue.size(); ++i) {
            double error =sqrt((predicted[i] - trueValue[i])*(predicted[i] - trueValue[i]));
            mse += error * error;
            delMSE[i] = 2.0 * error;
            std::cout << "delmse: " << delMSE[i] << "\t" << mse << "\t" << predicted[i] << "\t" << trueValue[i] << "\t" << error << std::endl;
        }
        
        mse /= trueValue.size();
        
        return {mse, delMSE};
    }

    // Function to calculate Binary Cross-Entropy and its gradient
    std::pair<double, std::vector<double>> calculateBCEandGrad(const std::vector<double>& trueValue, const std::vector<double>& predicted) {
        if (trueValue.size() != predicted.size()) {
            std::cerr << "Error: The size of trueValue and predicted arrays must be the same!" << std::endl;
            return {0.0, {}};
        }
        
        double bce = 0.0;
        std::vector<double> delBCE(predicted.size(), 0.0);
        
        for (size_t i = 0; i < trueValue.size(); ++i) {
            double y = trueValue[i];
            double yHat = predicted[i];
            
            yHat = std::max(1e-15, std::min(1.0 - 1e-15, yHat));
            
            bce -= (y * std::log(yHat+1e-15) + (1 - y) * std::log(1 - yHat+1e-15));
            
            delBCE[i]=-(y/(yHat+1e-2)-(1-y)/(1-yHat+1e-2));
            //std::cout<<predicted[i]<<",";
        }
        //std::cout<<std::endl;
        bce /= trueValue.size();
        //std::cout << "delmse: " << delBCE[0] << "\t" << bce << "\t" << predicted[0] << "\t" << trueValue[0] << "\t"  << std::endl;

        
        return {bce, delBCE};
    }
};

#endif // LOSS_H
