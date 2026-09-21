//
// Created by timwe on 10/31/2025.
//

#include "NeuralNetwork.h"

#include <chrono>

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

    template<Math::floatTypes T>
    Math::Matrix<T> NeuralNetwork<T>::forward(const Math::Matrix<T> &X) {
        if(this->layers.size() < 1)
            throw std::logic_error("Not enough layers in the Network");

        if(X.rows() != this->layers.front().getinNodes())
            throw std::logic_error("Input data does not match first layer shape");

        auto A = this->layers[0].forward(X);

        for(std::size_t i = 1; i < this->layers.size(); i++) {
            A = this->layers[i].forward(A);
        }

        return A; // Y-Hat from last layer shape (n_L x m)
    }

    template<Math::floatTypes T>
    void NeuralNetwork<T>::backward(const Math::Matrix<T> &Y, const Math::Matrix<T> &Yhat) {
        if(Y.rows() != Yhat.rows() || Y.cols() != Yhat.cols())
            throw std::logic_error("Y and Yhat shapes are not matching");

        if(this->layers.size() == 0)
            throw std::logic_error("Layer count is zero");

        if(Y.cols() == 0)
            throw std::logic_error("Y columns can't be zero");

        // auto m = static_cast<T>(Y.cols());
        Math::Matrix<T> dALast;

        if(this->loss == LossType::MSE) {
            dALast = (Yhat.sub(Y)).scalarMul(T{2}); // TODO: Check if it was correct to remove /m (reason being layer does also /m so it would become m^2)
        } else { //BCE
            auto onesSizeY = Y;
            onesSizeY.fill(T{1});
            auto p = Yhat.clip(1e-7);
            auto onesSizep = p;
            onesSizep.fill(T{1});

            auto term1 = Y.scalarMul(-1).divide(p);
            auto term2 = onesSizeY.sub(Y).divide(onesSizep.sub(p));

            dALast = (term1.add(term2)).scalarMul((T{1})); //TODO: Same check for /m as in MSE
        }

        Math::Matrix<T> dA;

        if(this->loss == LossType::BCE && this->layers.back().getActivation() == ActivationTypes::Sigmoid) {
            Math::Matrix<T> dZLast = layers.back().getA().sub(Y);
            dA = layers.back().backward(dZLast, true);
        } else {
            dA = layers.back().backward(dALast);
        }

        for(int i =this->layers.size()-2; i >= 0; i--) {
            dA = layers[i].backward(dA);
        }
    }

    // TODO: Implement batching
    template<Math::floatTypes T>
    T NeuralNetwork<T>::train(const Math::Matrix<T> &X, const Math::Matrix<T> &Y, bool timeExecution, bool printLoss, std::size_t printLossEveryXEpoch, bool exportLoss, std::size_t exportLossEveryXEpoch) {
        auto startTime = std::chrono::high_resolution_clock::now();

        // std::vector<size_t> idx(X.rows());
        // std::iota(std::begin(idx), std::end(idx), 0);

        for(std::size_t epoch = 0; epoch < this->epochs; epoch++) {
            auto Yhat = this->forward(X);
            if(printLoss)
                if(epoch % printLossEveryXEpoch == 0)
                    std::printf("Loss: %lf \n", this->compute_loss(Y, Yhat));

            this->backward(Y, Yhat);
            this->update();
        }

        if(timeExecution)
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime) << std::endl;

        auto Yhat = this->forward(X);
        return this->compute_loss(Y, Yhat);
    }

    // TODO: Should I shuffle the Data first before splitting?
    template<Math::floatTypes T>
    std::tuple<Math::Matrix<T>, Math::Matrix<T>, Math::Matrix<T>, Math::Matrix<T>> NeuralNetwork<T>::trainTestSplit(
        const Math::Matrix<T> &X, const Math::Matrix<T> &Y, const float trainSizeFloat) {

        const std::size_t N = X.cols();

        auto endTrain = std::floor(trainSizeFloat * static_cast<float>(N));

        auto startTest = endTrain + 1;
        auto endTest = N;

        Math::Matrix<T> XTrain(X.rows(), endTrain, 0);
        Math::Matrix<T> YTrain(Y.rows(), endTrain, 0);

        Math::Matrix<T> XTest(X.rows(), endTest - startTest, 0);
        Math::Matrix<T> YTest(Y.rows(), endTest - startTest, 0);

        for(int i = 0; i < endTrain; i++) {
            for(int j = 0; j < X.rows(); j++) {
                XTrain(j, i) = X(j, i);
            }
            for(int j = 0; j < Y.rows(); j++) {
                YTrain(j, i) = Y(j, i);
            }
        }

        for(int i = startTest; i < endTest; i++) {
            for(int j = 0; j < X.rows(); j++) {
                XTest(j, i - startTest) = X(j, i);
            }
            for(int j = 0; j < Y.rows(); j++) {
                YTest(j, i - startTest) = Y(j, i);
            }
        }

        return {XTrain, YTrain, XTest, YTest};
    }

    template <Math::floatTypes T>
    void NeuralNetwork<T>::inplaceScaleFeature(std::size_t featureIndex, Math::Matrix<T> &X, ScalerType scaler) {
        if(featureIndex >= X.rows())
            throw std::logic_error("Feature index out of bounds");

        if(scaler == ScalerType::minMax) {
            // TODO: Implement
            throw std::logic_error("MinMax Scaler is not implemented");
        } else if(scaler == ScalerType::robust) {
            // TODO: Implement
            throw std::logic_error("Robust Scaler is not implemented");
        } else if(scaler == ScalerType::zScore) {
            T meanVal = X.meanOfRow(featureIndex);
            T stdDevVal = X.stdDevOfRow(featureIndex);

            for(std::size_t i = 0; i < X.cols(); i++) {
                X(featureIndex, i) = (X(featureIndex, i) - meanVal) / stdDevVal;
            }
        } else {
            throw std::logic_error("Scaler type unkown");
        }
    }

    template<Math::floatTypes T>
    T NeuralNetwork<T>::compute_loss(const Math::Matrix<T> &Y, const Math::Matrix<T> &Yhat) {
        if(Y.rows() != Yhat.rows() || Y.cols() != Yhat.cols())
            throw std::logic_error("Y and Yhat shapes do not match");

        T lossVal;
        if(this->loss == LossType::MSE) {
            auto tempMatrix = Yhat.sub(Y);
            lossVal = tempMatrix.hadamard(tempMatrix).mean(); // mean((yhat-Y)^2)
        } else if(this->loss == LossType::BCE) {
            auto p = Yhat.clip(T{1e-7}); // 1e-7 from chatgpt
            auto part1 = Y.hadamard(p.log(std::exp(1.0)));

            auto onesSizeY = Y;
            onesSizeY.fill((T{1}));

            auto onesSizep = p;
            onesSizep.fill((T{1}));

            auto part2 = (onesSizeY.sub(Y)).hadamard((onesSizep.sub(p)).log(std::exp(1.0)));

            lossVal = -(part1.add(part2).mean());
        } else {
            lossVal = -std::numeric_limits<T>::infinity(); // Might be UB based on compiler flags :(
        }

        return lossVal;
    }

    template<Math::floatTypes T>
    void NeuralNetwork<T>::update() {
        for(auto& layer : this->layers) {
            layer.update(this->learningRate);
        }
    }

    template class NeuralNetworks::NeuralNetwork<float>;
    template class NeuralNetworks::NeuralNetwork<double>;
}
