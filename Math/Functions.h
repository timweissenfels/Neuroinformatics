//
// Created by timwe on 10/17/2025.
//

#ifndef NEUROINFORMATICS_FUNCTIONS_H
#define NEUROINFORMATICS_FUNCTIONS_H

#include <cmath>

namespace Math::Functions {
    template <class T, class F>
    [[nodiscard]] T log(F base, T num);

    template <class T>
    [[nodiscard]] T tanh(T num);

    template <class T>
    T sigmoid(T num);

    template <class T>
    T relu(T num);

    template <class T>
    T softplus(T num);

    template <class T>
    T mish(T num);

    template <class T>
    T delu(T num, int a = 1, int b = 2, double xc = 1.25643);

    template <class T, class F>
    T elu(T num, F alpha);
}

#endif //NEUROINFORMATICS_FUNCTIONS_H
