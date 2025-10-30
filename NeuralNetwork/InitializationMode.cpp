//
// Created by timwe on 10/30/2025.
//

#include "InitializationMode.h"

NeuralNetwork::InitializationMode getInitializationModeFromActivationFunction(NeuralNetwork::ActivationTypes act) {
    if(act == NeuralNetwork::ActivationTypes::Relu)
        return NeuralNetwork::InitializationMode::He;

    if(act == NeuralNetwork::ActivationTypes::Sigmoid || act == NeuralNetwork::ActivationTypes::Tanh)
        return NeuralNetwork::InitializationMode::Xavier;

    if(act == NeuralNetwork::ActivationTypes::Selu)
        return NeuralNetwork::InitializationMode::Lecun;

    return NeuralNetwork::InitializationMode::Xavier;
}
