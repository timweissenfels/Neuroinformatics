cmake_minimum_required(VERSION 3.30.5)
project(Neuroinformatics)

set(CMAKE_CXX_STANDARD 23)

add_executable(Neuroinformatics main.cpp
        Math/Matrix.cpp
        Math/Matrix.h
        Math/Functions.h
        Math/Functions.cpp
        NeuralNetworks/DenseLayer.cpp
        NeuralNetworks/DenseLayer.h
        NeuralNetworks/ActivationTypes.h
        NeuralNetworks/InitializationMode.cpp
        NeuralNetworks/InitializationMode.h
        NeuralNetworks/NeuralNetwork.cpp
        NeuralNetworks/NeuralNetwork.h
        NeuralNetworks/LossType.h)
