#include <iostream>
#include <cmath>
#include <chrono>

#include "NeuralNetworks/NeuralNetwork.h"
#include "NeuralNetworks/LossType.h"
#include "Math/Matrix.h"

void sinPOC() {
    auto X = Math::Matrix<float>(1,71,0);
    auto Y = Math::Matrix<float>(1,71,0);

    for(int i = 0; i <= 70; i++) {
        X(0, i) = (float)i/10;
        Y(0,i) = sinf((float)i/10);
    }

    NeuralNetworks::NeuralNetwork sinNN(NeuralNetworks::LossType::MSE, (double)0.09, 1000, 32, 42, X);
    sinNN.AddDenseLayer(1,4,NeuralNetworks::ActivationTypes::Tanh);
    sinNN.AddDenseLayer(4,1,NeuralNetworks::ActivationTypes::Linear);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < 5000; ++step) {
        auto YHat = sinNN.forward(X);
        // float loss = sinNN.compute_loss(Y, YHat);
        // if (step % 50 == 0) std::cout << "step " << step << " loss " << loss << "\n";
        sinNN.backward(Y, YHat);
        sinNN.update();
    }

    auto YHat = sinNN.forward(X);
    std::cout << "final loss " << sinNN.compute_loss(Y, YHat) << "\n";
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
}

void xorPOC() {
    auto X = Math::Matrix<float>(2, 4);
    auto Y = Math::Matrix<float>(1, 4);

    X(0,0) = 0;
    X(1,0) = 0;
    Y(0,0) = 0;

    X(0,1) = 0;
    X(1,1) = 1;
    Y(0,1) = 1;

    X(0,2) = 1;
    X(1,2) = 0;
    Y(0,2) = 1;

    X(0,3) = 1;
    X(1,3) = 1;
    Y(0,3) = 0;

    NeuralNetworks::NeuralNetwork xorNN(NeuralNetworks::LossType::BCE, (double)0.05, 1000, 32, 42, X);
    xorNN.AddDenseLayer(2,8,NeuralNetworks::ActivationTypes::ReLU);
    xorNN.AddDenseLayer(8,1,NeuralNetworks::ActivationTypes::Sigmoid);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < 5000; ++step) {
        auto YHat = xorNN.forward(X);
        float loss = xorNN.compute_loss(Y, YHat);
        if (step % 50 == 0) std::cout << "step " << step << " loss " << loss << "\n";
        xorNN.backward(Y, YHat);
        xorNN.update();
    }

    auto YHat = xorNN.forward(X);
    std::cout << "final loss " << xorNN.compute_loss(Y, YHat) << "\n";
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
}

int main() {
    xorPOC();
    return 0;
}