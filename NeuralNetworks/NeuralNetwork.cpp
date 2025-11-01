//
// Created by timwe on 10/31/2025.
//

#include "NeuralNetwork.h"

namespace NeuralNetworks {
    template<Math::floatTypes T>
    NeuralNetwork<T>::NeuralNetwork(LossType _loss, double _learningRate, std::size_t _epochs, std::size_t _batchSize, std::size_t rngSeed, Math::Matrix<T> &_data)
        : loss(_loss), learningRate(_learningRate), epochs(_epochs), batchSize(_batchSize),  data(_data) {
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

        auto dA = layers.back().backward(dALast);
        for(int i = this->layers.size()-2; i >= 0; i--) {
            dA = layers[i].backward(dA);
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
