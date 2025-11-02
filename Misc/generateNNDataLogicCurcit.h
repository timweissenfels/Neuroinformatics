//
// Created by timwe on 10/20/2025.
//

#include <iostream>
#include <vector>
#include <bitset>
#include <functional>
#include <utility>

// Output Bitset could also be a tuple tbh
template <std::size_t T, std::size_t F>
std::vector<std::pair<std::bitset<T>, std::bitset<F>>> generateNNData(std::function<std::bitset<F>(const std::bitset<T>)> mapper) {

    std::vector<std::pair<std::bitset<T>, std::bitset<F>>> returnVector(1 << T); // (1 << T) == 2^T

    for(std::size_t i = 0; i < (1 << T); i++) {
        auto input = std::bitset<T>(i);
        auto output = mapper(input);
        returnVector.at(i) = {input, output};
    }

    return returnVector;
}


