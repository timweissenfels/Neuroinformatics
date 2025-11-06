#include <iostream>
#include <cmath>
#include <chrono>

#include "NeuralNetworks/NeuralNetwork.h"
#include "NeuralNetworks/LossType.h"
#include "Math/Matrix.h"
#include "Misc/generateNNDataLogicCurcit.h"
#include "Data/readHousingData.h"
#include "NeuralNetworks/ScalerType.h"

void sinPOC() {
    auto X = Math::Matrix<float>(1,701,0);
    auto Y = Math::Matrix<float>(1,701,0);

    for(int i = 0; i <= 700; i++) {
        X(0, i) = static_cast<float>(i)/100;
        Y(0,i) = sinf(static_cast<float>(i)/100) + cosf(static_cast<float>(i)/100);
    }

    NeuralNetworks::NeuralNetwork<float> sinNN(NeuralNetworks::LossType::MSE, 0.09, 5000, 32, 42);
    sinNN.AddDenseLayer(1,16,NeuralNetworks::ActivationTypes::Tanh);
    sinNN.AddDenseLayer(16,16,NeuralNetworks::ActivationTypes::Tanh);
    sinNN.AddDenseLayer(16,1,NeuralNetworks::ActivationTypes::Linear);

    auto [XTrain, YTrain, XTest, YTest] = sinNN.trainTestSplit(X, Y, 0.8f);

    auto finalLoss = sinNN.train(XTrain, YTrain, true, true, 1000);

    auto test_loss = sinNN.forward(XTest);

    std::cout << "Test Loss: " << sinNN.compute_loss(YTest, test_loss) << "\n";

    std::cout << "final loss " << finalLoss << "\n";
}

void xorPOC() {
    auto X = Math::Matrix<float>(2, 4);
    auto Y = Math::Matrix<float>(1, 4);

    X(0,0) = 0;
    X(1,0) = 0;
    Y(0,0) = 0;

    X(0,1) = 0;
    X(1,1) = 1;
    Y(0,1) = 1;

    X(0,2) = 1;
    X(1,2) = 0;
    Y(0,2) = 1;

    X(0,3) = 1;
    X(1,3) = 1;
    Y(0,3) = 0;

    NeuralNetworks::NeuralNetwork<float> xorNN(NeuralNetworks::LossType::BCE, (double)0.05, 5000, 32, 42);
    xorNN.AddDenseLayer(2,8,NeuralNetworks::ActivationTypes::ReLU);
    xorNN.AddDenseLayer(8,1,NeuralNetworks::ActivationTypes::Sigmoid);

    auto finalLoss = xorNN.train(X, Y, true, true, 1000);
    std::cout << "final loss " << finalLoss << "\n";
}

std::pair<Math::Matrix<float>, Math::Matrix<float>> generateNNDataLogicCurcit() {
    constexpr unsigned short inputShape = 3;
    constexpr unsigned short outputShape = 1;

    Math::Matrix<float> X(3,std::pow(2, 3));
    Math::Matrix<float> Y(1,std::pow(2, 3));

    auto mapper = [](const std::bitset<inputShape>& input) -> std::bitset<outputShape> {
        std::bitset<1> result;
        auto term1 = !input[0] && !input[1] && !input[2];
        auto term2 = !input[0] && input[1] && !input[2];
        auto term3 = input[2] && input[1] && !input[0];
        auto term4 = input[0] && input[1] && input[2];

        result[0] =  term1 || term2 || term3 || term4;

        return result;
    };

    auto data = generateNNData<inputShape, outputShape>(mapper);

    for (std::size_t i = 0; i < data.size(); i++) {
        std::cout << "Input: " << data.at(i).first << " - Output: " << data.at(i).second << '\n';

        X(0,i) = static_cast<float>(data.at(i).first[0]);
        X(1,i) = static_cast<float>(data.at(i).first[1]);
        X(2,i) = static_cast<float>(data.at(i).first[2]);
        Y(0,i) = data.at(i).second.to_ulong();
    }
    return {X,Y};
}

void logicPOC() {
    const auto& [X, Y] = generateNNDataLogicCurcit();

    NeuralNetworks::NeuralNetwork<float> xorNN(NeuralNetworks::LossType::BCE, 0.05, 5000, 32, 42);
    xorNN.AddDenseLayer(3,8,NeuralNetworks::ActivationTypes::Tanh);
    xorNN.AddDenseLayer(8,8,NeuralNetworks::ActivationTypes::Tanh);
    xorNN.AddDenseLayer(8,1,NeuralNetworks::ActivationTypes::Sigmoid);

    auto finalLoss = xorNN.train(X, Y, true, true, 5000);

    std::cout << "final loss " << finalLoss << "\n";
}

void housingPOC() {
    auto data = readHousingData("C:\\Users\\UI703201\\Desktop\\Neuroinformatics\\Data\\housing.csv");
    const auto& [X, Y] = getXandYVectors(data);


    NeuralNetworks::NeuralNetwork<float> housingNN(NeuralNetworks::LossType::MSE, 0.001, 500, 32, 42);
    auto [XTrain, YTrain, XTest, YTest] = housingNN.trainTestSplit(X, Y, 0.8f);


    // Scalings are fucked up, stddev and mean should be calculated on training data only and then applied to test set
    for(int i = 3; i < X.rows()-1; i++) {
        XTrain.log1pInplaceOfRow(i);
        XTest.log1pInplaceOfRow(i);
    }

    for(int i = 0; i < X.rows(); i++) {
        NeuralNetworks::NeuralNetwork<float>::inplaceScaleFeature(i, XTrain, NeuralNetworks::ScalerType::zScore);
        NeuralNetworks::NeuralNetwork<float>::inplaceScaleFeature(i, XTest, NeuralNetworks::ScalerType::zScore);
    }

    YTrain.log1pInplaceOfRow(0);
    NeuralNetworks::NeuralNetwork<float>::inplaceScaleFeature(0, YTrain, NeuralNetworks::ScalerType::zScore);

    YTest.log1pInplaceOfRow(0);
    NeuralNetworks::NeuralNetwork<float>::inplaceScaleFeature(0, YTest, NeuralNetworks::ScalerType::zScore);

    housingNN.AddDenseLayer(12,128,NeuralNetworks::ActivationTypes::Tanh);
    housingNN.AddDenseLayer(128,64,NeuralNetworks::ActivationTypes::Tanh);
    housingNN.AddDenseLayer(64,1,NeuralNetworks::ActivationTypes::Linear);

    auto finalLoss = housingNN.train(XTrain, YTrain, true, true, 100);

    auto test_loss = housingNN.forward(XTest);

    std::cout << "Test Loss: " << housingNN.compute_loss(YTest, test_loss) << "\n";

    std::cout << "final loss " << finalLoss << "\n";
}

int main() {
    // sinPOC();
    // xorPOC();
    // logicPOC();
    housingPOC();

    return 0;
}