//
// Created by timwe on 10/29/2025.
//

#ifndef NEUROINFORMATICS_DENSELAYER_H
#define NEUROINFORMATICS_DENSELAYER_H

#include <iostream>
#include "../Math/Matrix.h"
#include "ActivationTypes.h"
#include "InitializationMode.h"

namespace NeuralNetwork {
    template<Math::floatTypes T>
    class DenseLayer {
    private:
        std::size_t inNodes; // Number of nodes from last layer
        std::size_t outNodes; // Number of nodes in this layer
        //std::size_t m; // Batch size !!! Should be taken from Aprev.cols()
        Math::Matrix<T> W; // Weights; Shape (outNodes x inNodes)
        Math::Matrix<T> b; // Biases; Shape (outNodes x 1) only one per Node
        Math::Matrix<T> Z; // Cache for backprop, pre activation; Shape (outNodes x m); Z = W * Aprev + b
        Math::Matrix<T> dW; // Shape (outNodes x inNodes)
        Math::Matrix<T> db; // Shape (outNodes x 1)
        Math::Matrix<T> Aprev; // Cache for backprop, input to this layer; Shape (inNodes x m)
        NeuralNetwork::ActivationTypes act; // Activation function
        InitializationMode initMode;

    public:
        DenseLayer(std::size_t _inNodes, std::size_t _outNodes, NeuralNetwork::ActivationTypes _act); // Constructor
        [[nodiscard]] Math::Matrix<T> forward(const Math::Matrix<T>& Aprev); // Returns A, stores Aprev and Z
        [[nodiscard]] Math::Matrix<T> backward(const Math::Matrix<T>& dA); // Returns dA_prev; also computs dW, db stored  internally for updated
        void update(T lr); //SGD step: W -= learningRate*dW; b -= learningRate*db
        void initialize();
    };



} // NeuralNetwork

#endif //NEUROINFORMATICS_DENSELAYER_H
