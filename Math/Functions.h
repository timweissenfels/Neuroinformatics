//
// Created by timwe on 10/17/2025.
//

#include <iostream>
#include <algorithm>
#include <cmath>

namespace Math::Functions {
    template <class T, class F>
    T log(F base, T num) {
       return std::log(num) / std::log(base);
    }

    template <class T>
    T tanh(T num) {
        return (std::exp(num) - std::exp(-num)) / (std::exp(num) + std::exp(-num)); //Okay to use std::tanh?
    }

    // TODO: Check if its numerically stable for large |Z|
    template <class T>
    T sigmoid(T num) {
        return 1 / (1 + std::exp(-num)); // Logistic sigmoid
    }

    template <class T>
    T relu(T num) {
        return std::max(T(0), num); // (x+|x|)/2 without std::max function implementation
    }

    template <class T>
    T softplus(T num) {
        return log(std::exp(1.0), 1+std::exp(num)); // exp(1.0) to get e as the pre-defined base
    }

    template <class T>
    T mish(T num) {
        return num * tanh(softplus(num));
    }

    template <class T>
    T linear(T num) {
        return num;
    }

    template <class T>
    T delu(T num, int a = 1, int b = 2, double xc = 1.25643) { // https://en.wikipedia.org/wiki/Rectified_linear_unit#DELU
        if(b != 0)
            throw std::invalid_argument("Invalid hyperparameter b for delu, has to be unequal to 0");

        return (num > xc) ? num : (std::exp(a*num)-1)/b;
    }

    template <class T, class F>
    T elu(T num, F alpha) { // https://en.wikipedia.org/wiki/Rectified_linear_unit#ELU
        if(alpha < 0)
            throw std::invalid_argument("Invalid hyperparameter alpha for elu, has to be bigger than 0");

        return (num > 0) ? num : alpha*(std::exp(num)-1);
    }

    template <class T>
    T clip(const T num, const double epsilon = 1e-7) { // https://stackoverflow.com/a/9324086
        return std::max(epsilon, std::min(num, T{1}-epsilon));
    }

    template <class T>
    T clamp(const T num, const T epsilon = 1e-7) {
        return std::clamp(num, epsilon, T{1} - epsilon);
    }
}