//
// Created by timwe on 10/31/2025.
//

#ifndef NEUROINFORMATICS_NEURALNETWORK_H
#define NEUROINFORMATICS_NEURALNETWORK_H

#include "../Math/Matrix.h"
#include "ActivationTypes.h"
#include "DenseLayer.h"
#include "LossType.h"

namespace NeuralNetworks {
    template<Math::floatTypes T>
    class NeuralNetwork {
    private:
        std::vector<DenseLayer<T>> layers;
        LossType loss;
        double learningRate;
        std::size_t epochs;
        std::size_t batchSize;
        std::mt19937 gen;

    public:
        explicit NeuralNetwork(LossType _loss, double _learningRate, std::size_t _epochs, std::size_t _batchSize, std::size_t rngSeed);

        void AddDenseLayer(std::size_t inNodes, std::size_t outNodes, ActivationTypes act, bool initializeConstructor = true);
    };

    extern template class NeuralNetworks::NeuralNetwork<float>;
    extern template class NeuralNetworks::NeuralNetwork<double>;
}

#endif //NEUROINFORMATICS_NEURALNETWORK_H
