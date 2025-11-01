#include <iostream>
#include <cmath>
#include <chrono>
#include "NeuralNetworks/NeuralNetwork.h"
#include "NeuralNetworks/LossType.h"
#include "Math/Matrix.h"

int main() {
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
    return 0;
}