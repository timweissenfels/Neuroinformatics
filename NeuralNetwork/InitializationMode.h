//
// Created by timwe on 10/30/2025.
//

#ifndef NEUROINFORMATICS_INITIALIZATIONMODE_H
#define NEUROINFORMATICS_INITIALIZATIONMODE_H

#include "ActivationTypes.h"

namespace NeuralNetwork {
    enum class InitializationMode {
        He, Xavier, Lecun
    };

    InitializationMode getInitializationModeFromActivationFunction(NeuralNetwork::ActivationTypes act);
}

#endif //NEUROINFORMATICS_INITIALIZATIONMODE_H
