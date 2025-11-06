//
// Created by timwe on 10/31/2025.
//

#ifndef NEUROINFORMATICS_NEURALNETWORK_H
#define NEUROINFORMATICS_NEURALNETWORK_H

#include "../Math/Matrix.h"
#include "ActivationTypes.h"
#include "DenseLayer.h"
#include "LossType.h"
#include "ScalerType.h"

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
        explicit NeuralNetwork(LossType _loss, double _learningRate, std::size_t _epochs, std::size_t _batchSize,
                               std::size_t rngSeed);

        void AddDenseLayer(std::size_t inNodes, std::size_t outNodes, ActivationTypes act,
                           bool initializeConstructor = true);

        Math::Matrix<T> forward(const Math::Matrix<T> &X); // X -> shape (n_0 x m)

        T compute_loss(const Math::Matrix<T>& Y, const Math::Matrix<T>& Yhat);
        void backward(const Math::Matrix<T>& Y, const Math::Matrix<T>& Yhat);
        T train(const Math::Matrix<T>& X, const Math::Matrix<T>& Y, bool timeExecution = false, bool printLoss = false, std::size_t printLossEveryXEpoch = 50, bool exportLoss = false, std::size_t exportLossEveryXEpoch = 50);
        std::tuple<Math::Matrix<T>, Math::Matrix<T>, Math::Matrix<T>, Math::Matrix<T>> trainTestSplit(const Math::Matrix<T>& X, const Math::Matrix<T>& Y, float trainSizeFloat);

        static void inplaceScaleFeature(std::size_t featureIndex, Math::Matrix<T> &X, ScalerType scaler = ScalerType::zScore);

        void update();

    };

    extern template
    class NeuralNetworks::NeuralNetwork<float>;

    extern template
    class NeuralNetworks::NeuralNetwork<double>;
}

#endif //NEUROINFORMATICS_NEURALNETWORK_H
