//
// Created by timwe on 10/20/2025.
//

#include <iostream>
#include <vector>
#include <bitset>
#include <functional>
#include <utility>

// Output Bittset could also be a tuple tbh
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

int main() {
    constexpr unsigned short inputShape = 2;
    constexpr unsigned short outputShape = 1;

    auto mapper = [](const std::bitset<inputShape>& input) -> std::bitset<outputShape> {
        std::bitset<1> result;
        result[0] = input[0] && input[1];
        return result;
    };

    auto data = generateNNData<inputShape, outputShape>(mapper);

    for (const auto& pair : data) {
        std::cout << "Input: " << pair.first << " - Output: " << pair.second << '\n';
    }
    return 0;
}
