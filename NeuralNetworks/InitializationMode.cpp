//
// Created by timwe on 10/30/2025.
//

#include "InitializationMode.h"

namespace NeuralNetworks {
    NeuralNetworks::InitializationMode
    getInitializationModeFromActivationFunction(NeuralNetworks::ActivationTypes act) {
        if (act == NeuralNetworks::ActivationTypes::ReLU || act == NeuralNetworks::ActivationTypes::LeakyReLU)
            return NeuralNetworks::InitializationMode::He;

        if (act == NeuralNetworks::ActivationTypes::Sigmoid || act == NeuralNetworks::ActivationTypes::Tanh)
            return NeuralNetworks::InitializationMode::Xavier;

        if (act == NeuralNetworks::ActivationTypes::SELU)
            return NeuralNetworks::InitializationMode::Lecun;

        return NeuralNetworks::InitializationMode::Xavier;
    }
}