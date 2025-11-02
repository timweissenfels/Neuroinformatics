//
// Created by timwe on 10/29/2025.
//

#include <cmath>
#include "DenseLayer.h"

namespace NeuralNetworks {

    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::linearDerivative(const std::size_t _rows, const std::size_t _cols, const std::size_t _stride) const {
        Math::Matrix<T> ones(_rows, _cols, _stride);
        ones.fill(T{1});

        return ones;
    }

    // A-based
    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::sigmoidDerivative() const {
        Math::Matrix<T> ones(this->A.rows(), this->A.cols(), this->A.stride());
        ones.fill(T{1});

        return this->A.hadamard(ones.sub(this->A));
    }

    // A-based
    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::tanhDerivative() const {
        Math::Matrix<T> ones(this->A.rows(), this->A.cols(), this->A.stride());
        ones.fill(T{1});

        return ones.sub(this->A.hadamard(this->A));
    }

    // Z-based
    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::reluDerivative() const {
        return this->Z.map([](T num){ return num > T{0} ? T{1} : T{0}; });
    }

    // Z-based
    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::eluDerivative(T alpha) const {
        return this->Z.map([&](T num){ return num > T{0} ? T{1} : alpha*std::exp(num); });
    }

    // Z-based
    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::softplusDerivative() const {
        return this->Z.sigmoid();
    }

    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::mishDerivative() const {
        throw std::logic_error("Not implemented yet");
    }

    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::deluDerivative() const {
        throw std::logic_error("Not implemented yet");
    }

    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::applyActivation(const Math::Matrix<T> &mat) const {
        if(this->act == NeuralNetworks::ActivationTypes::Tanh)
            return mat.tanh();
        else if(this->act == NeuralNetworks::ActivationTypes::ReLU)
            return mat.relu();
        else if(this->act == NeuralNetworks::ActivationTypes::Sigmoid)
            return mat.sigmoid();
        else if(this->act == NeuralNetworks::ActivationTypes::Softplus)
            return mat.softplus();
        else if(this->act == NeuralNetworks::ActivationTypes::Elu)
            return mat.elu(0.5);
        else if(this->act == NeuralNetworks::ActivationTypes::Delu)
            return mat.delu();
        else if(this->act == NeuralNetworks::ActivationTypes::Mish)
           return mat.mish();
        else if(this->act == NeuralNetworks::ActivationTypes::Linear)
            return mat.linear();
        else
           return mat.tanh();
    }

    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::applyDerivative() const {
        if(this->act == NeuralNetworks::ActivationTypes::Linear)
            return this->linearDerivative(A.rows(), A.cols(), A.stride());
        else if(this->act == NeuralNetworks::ActivationTypes::Tanh)
            return this->tanhDerivative();
        else if(this->act == NeuralNetworks::ActivationTypes::ReLU)
            return this->reluDerivative();
        else if(this->act == NeuralNetworks::ActivationTypes::Sigmoid)
            return this->sigmoidDerivative();
        else if(this->act == NeuralNetworks::ActivationTypes::Softplus)
            return this->softplusDerivative();
        else if(this->act == NeuralNetworks::ActivationTypes::Elu)
            return this->eluDerivative(0.5);
        else if(this->act == NeuralNetworks::ActivationTypes::Delu)
            return this->deluDerivative();
        else if(this->act == NeuralNetworks::ActivationTypes::Mish)
            return this->mishDerivative();
        else
            throw std::logic_error("Derivative type is unknown in DenseLayer::applyDerivative");
    }

    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::forward(const Math::Matrix<T> &_Aprev) {
        if(_Aprev.rows() != this->inNodes)
            throw std::invalid_argument("Aprev has an unexpected amount of features");

        if(W.rows() != this->outNodes || W.cols() != this->inNodes)
            throw std::invalid_argument("W has an unexpected shape");

        if(b.rows() != this->outNodes || b.cols() != 1)
            throw std::invalid_argument("b has an unexpected shape");

        // Store Aprev (in x m) and Z ( out x m) after forward
        this->Z = (this->W.matMul(_Aprev)).addBias(this->b);
        this->Aprev = _Aprev;
        this->A = applyActivation(this->Z);

        return this->A;
    }

    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::backward(const Math::Matrix<T> &dA, bool treatInputASdZ) {
        if(dA.rows() != this->outNodes)
            throw std::invalid_argument("dA Shape is not matching features of the layer");

        if(this->Aprev.rows() != this->inNodes)
            throw std::invalid_argument("Aprev Shape is not matching features of the input layer");

        if(this->Aprev.cols() == 0 || this->Aprev.cols() != this->Z.cols())
            throw std::invalid_argument("Aprev columns are 0 or shape is not matching Z");

        if(dA.cols() != this->Z.cols())
            throw std::invalid_argument("dA upstream passes does not match the Z batch size");

        const T m = this->Aprev.cols();

        if(treatInputASdZ) // BCE + Sigmoid trick
            this->dZ = dA;
        else
            this->dZ = dA.hadamard(applyDerivative());

        this->dW = (this->dZ.matMul(this->Aprev.transpose())).divide(m);
        this->db = dZ.sumOverColumns().divide(m);
        return this->W.transpose().matMul(this->dZ); // Return dAprev
    }

    template<Math::floatTypes T>
    void DenseLayer<T>::update(T lr) {
        this->W = this->W.sub(this->dW.scalarMul(lr));
        this->b = this->b.sub(this->db.scalarMul(lr));
    }

    template<Math::floatTypes T>
    ActivationTypes DenseLayer<T>::getActivation() noexcept {
        return this->act;
    }

    template<Math::floatTypes T>
    Math::Matrix<T> DenseLayer<T>::getA() noexcept {
        return this->A;
    }

    template<Math::floatTypes T>
    void DenseLayer<T>::xavierInitializer() {
        double sigma = std::sqrt(2.0/(this->inNodes + this->outNodes));
        std::normal_distribution<T> norm(0,sigma); // mean 0; stddev sigma

        this->W = this->W.map([&](T num){return T{norm(this->gen)};});
    }

    template<Math::floatTypes T>
    void DenseLayer<T>::heInitializer() {
        double sigma = std::sqrt(2.0/this->inNodes);
        std::normal_distribution<T> norm(0,sigma); // mean 0; stddev sigma

        this->W = this->W.map([&](T num){return T{norm(this->gen)};});
    }

    template<Math::floatTypes T>
    void DenseLayer<T>::lecunInitializer() {
        double sigma = std::sqrt(1.0/(this->inNodes));
        std::normal_distribution<T> norm(0,sigma); // mean 0; stddev sigma

        this->W = this->W.map([&](T num){return T{norm(this->gen)};});
    }

    template<Math::floatTypes T>
    void DenseLayer<T>::initialize() {
        if (this->initMode == NeuralNetworks::InitializationMode::Xavier)
            this->xavierInitializer();
        else if(this->initMode == NeuralNetworks::InitializationMode::He)
            this->heInitializer();
        else if(this->initMode == NeuralNetworks::InitializationMode::Lecun)
            this->lecunInitializer();
        else
            this->xavierInitializer();
    }

    template<Math::floatTypes T>
    std::size_t DenseLayer<T>::getinNodes() noexcept {
        return this->inNodes;
    }

    template<Math::floatTypes T>
    std::size_t DenseLayer<T>::getoutNodes() noexcept {
        return this->outNodes;
    }

    template<Math::floatTypes T>
    DenseLayer<T>::DenseLayer(std::size_t _inNodes, std::size_t _outNodes, NeuralNetworks::ActivationTypes _act, std::mt19937& _gen, bool initializeInConstructor)
        : gen(_gen), inNodes(_inNodes), outNodes(_outNodes), act(_act) {
        this->initMode = NeuralNetworks::getInitializationModeFromActivationFunction(this->act);
        this->W = Math::Matrix<T>(this->outNodes, this->inNodes);
        this->b = Math::Matrix<T>(this->outNodes, 1);
        this->dW = Math::Matrix<T>(this->outNodes, this->inNodes);
        this->db = Math::Matrix<T>(this->outNodes, 1);
        this->b.fill(0);

        if (initializeInConstructor)
            this->initialize();
    }

    template class NeuralNetworks::DenseLayer<float>;
    template class NeuralNetworks::DenseLayer<double>;
} // NeuralNetworks