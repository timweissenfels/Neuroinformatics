//
// Created by timwe on 10/29/2025.
//

#include "DenseLayer.h"

namespace NeuralNetwork {

    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::forward(const Math::Matrix<T> &Aprev) {
        return Math::Matrix<T>(0, 0);
    }

    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::backward(const Math::Matrix<T> &dA) {
        return Math::Matrix<T>(0, 0);
    }

    template<Math::floatTypes T>
    void DenseLayer<T>::update(T lr) {
    }

    template<Math::floatTypes T>
    void DenseLayer<T>::initialize() {
        if (this->initMode == NeuralNetwork::InitializationMode::Xavier)
            return;
        else if(this->initMode == NeuralNetwork::InitializationMode::He)
            return;
        else if(this->initMode == NeuralNetwork::InitializationMode::Lecun)
            return;
        else
            return;

    }

    template<Math::floatTypes T>
    DenseLayer<T>::DenseLayer(std::size_t _inNodes, std::size_t _outNodes, NeuralNetwork::ActivationTypes _act) {
        this->inNodes = _inNodes;
        this->outNodes = _outNodes;
        this->act = _act;
        this->initMode = NeuralNetwork::getInitializationModeFromActivationFunction(this->act);
        this->W = Math::Matrix<T>(this->outNodes, this->inNodes);
        this->b = Math::Matrix<T>(this->outNodes, 1);
        //this->Z = Math::Matrix<T>(this->outNodes, )



    }



} // NeuralNetwork