//
// Created by timwe on 10/29/2025.
//

#ifndef NEUROINFORMATICS_DENSELAYER_H
#define NEUROINFORMATICS_DENSELAYER_H

#include <iostream>
#include <random>
#include "../Math/Matrix.h"
#include "ActivationTypes.h"
#include "InitializationMode.h"

namespace NeuralNetworks {
    template<Math::floatTypes T>
    class DenseLayer {
    private:
        std::size_t inNodes; // Number of nodes from last layer
        std::size_t outNodes; // Number of nodes in this layer
        NeuralNetworks::ActivationTypes act; // Activation function
        InitializationMode initMode; // Initialization Mode picked based on the activation function
        std::mt19937& gen; // Random generator for norm distributions used for the weigths

        Math::Matrix<T> W; // Weights; Shape (outNodes x inNodes)
        Math::Matrix<T> b; // Biases; Shape (outNodes x 1) only one per Node
        Math::Matrix<T> Z; // Cache for backprop, pre activation; Shape (outNodes x m); Z = W * Aprev + b
        Math::Matrix<T> dZ; // Cache for inspection; Can be kept in backward function
        Math::Matrix<T> A; // Cache for A; Z after Activation; Shape (outNodes x m); A = activation(Z)
        // Math::Matrix<T> dAprev; // Cache for inspection; Is returned from backward function
        Math::Matrix<T> dW; // Shape (outNodes x inNodes)
        Math::Matrix<T> db; // Shape (outNodes x 1)
        Math::Matrix<T> Aprev; // Cache for backprop, input to this layer; Shape (inNodes x m)

        void xavierInitializer();
        void heInitializer();
        void lecunInitializer();
        Math::Matrix<T> applyActivation(const Math::Matrix<T>&) const;
        Math::Matrix<T> applyDerivative(bool = false) const;

        Math::Matrix<T> linearDerivative(std::size_t, std::size_t, std::size_t) const;
        Math::Matrix<T> sigmoidDerivative() const;
        Math::Matrix<T> tanhDerivative() const;
        Math::Matrix<T> reluDerivative() const;
        Math::Matrix<T> eluDerivative(T) const;
        Math::Matrix<T> softplusDerivative() const;
        [[maybe_unused]] Math::Matrix<T> mishDerivative() const;
        [[maybe_unused]] Math::Matrix<T> deluDerivative() const;

    public:
        explicit DenseLayer(std::size_t _inNodes, std::size_t _outNodes, NeuralNetworks::ActivationTypes _act, std::mt19937& gen, bool initializeInConstructor = true); // Constructor
        [[nodiscard]] Math::Matrix<T> forward(const Math::Matrix<T>& Aprev); // Returns A, stores Aprev and Z
        [[nodiscard]] Math::Matrix<T> backward(const Math::Matrix<T>& dA); // Returns dA_prev; also computs dW, db stored  internally for updated
        [[nodiscard]] std::size_t getinNodes() noexcept;
        [[nodiscard]] std::size_t getoutNodes() noexcept;

        void update(T lr); //SGD step: W -= learningRate*dW; b -= learningRate*db
        void initialize();
    };



    extern template class NeuralNetworks::DenseLayer<float>;
    extern template class NeuralNetworks::DenseLayer<double>;
} // NeuralNetworks

#endif //NEUROINFORMATICS_DENSELAYER_H
