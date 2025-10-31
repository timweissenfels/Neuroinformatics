//
// Created by timwe on 10/31/2025.
//

#include "NeuralNetwork.h"

namespace NeuralNetworks {
    template<Math::floatTypes T>
    NeuralNetwork<T>::NeuralNetwork(LossType _loss, double _learningRate, std::size_t _epochs, std::size_t _batchSize, std::size_t rngSeed)
        : loss(_loss), learningRate(_learningRate), epochs(_epochs), batchSize(_batchSize) {
        this->gen = std::mt19937 {static_cast<uint32_t>(rngSeed)};

    }

    template<Math::floatTypes T>
    void NeuralNetwork<T>::AddDenseLayer(std::size_t inNodes, std::size_t outNodes, ActivationTypes act, bool initializeConstructor) {
        if(!this->layers.empty())
            if(inNodes != this->layers.back().getoutNodes())
                throw std::logic_error("inNodes does not match outNodes of last layer");

        this->layers.emplace_back(inNodes, outNodes, act, this->gen, initializeConstructor);
    }

    template class NeuralNetworks::NeuralNetwork<float>;
    template class NeuralNetworks::NeuralNetwork<double>;
}
