cmake_minimum_required(VERSION 3.30.5)
project(Neuroinformatics)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "-O0 -ffast-math -Wall -march=native")

add_executable(Neuroinformatics main.cpp
        Math/Matrix.cpp
        Math/Matrix.h
        Math/Functions.h
        Math/Functions.h
        NeuralNetworks/DenseLayer.cpp
        NeuralNetworks/DenseLayer.h
        NeuralNetworks/ActivationTypes.h
        NeuralNetworks/InitializationMode.cpp
        NeuralNetworks/InitializationMode.h
        NeuralNetworks/NeuralNetwork.cpp
        NeuralNetworks/NeuralNetwork.h
        NeuralNetworks/LossType.h
        tests/Functions/Functions.h
        Data/readHousingData.h)

project(baz LANGUAGES CXX VERSION 0.0.1)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_executable(tests tests/test.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH catch2-src/extras)
include(CTest)
include(Catch)
catch_discover_tests(tests)