//
// Created by Tim Weissenfels on 10.10.25.
//

#include "Matrix.h"
#include "Functions.h"
#include <algorithm>
//#include <arm_neon.h>
#include <iostream>

namespace Math {

    template<floatTypes T>
    Matrix<T>::Matrix() noexcept : rows_(0), cols_(0), stride_(0) {
        stride_ = 0;  // round up

        this->data_.resize(rows_ * stride_);
    }

    template<floatTypes T>
    Matrix<T>::Matrix(std::size_t rows, std::size_t cols, std::size_t stride): rows_(rows), cols_(cols), stride_(stride) {
        if (rows_ == 0 || cols_ == 0)
            throw std::invalid_argument("Invalid rows and columns provided, one of them is 0");

        if (stride_ == 0) {
            // pad to 8 for float, 4 for double (AVX2); adjust as you like
            const std::size_t w = std::is_same_v<T,float> ? 8 : 4;
            stride_ = ((cols + w - 1) / w) * w;  // round up
        } else if (stride_ < cols_) {
            throw std::invalid_argument("stride must be bigger or equal to columns");
        }

        this->data_.resize(rows_ * stride_);
    }

    template<floatTypes T>
    T& Matrix<T>::operator()(std::size_t r, std::size_t c) {
        if (r >= rows_ || c >= cols_)
            throw std::out_of_range("In Matrix::operator() r or c are out of bounds");

        return this->data_[r * stride_ + c];
    }

    template<floatTypes T>
    const T& Matrix<T>::operator()(std::size_t r, std::size_t c) const {
        if (r >= rows_ || c >= cols_)
            throw std::out_of_range("In Matrix::operator() r or c are out of bounds");

        return this->data_[r * stride_ + c];
    }

    template<floatTypes T>
    std::span<T> Matrix<T>::data() noexcept {
        return this->data_;
    }

    template<floatTypes T>
    std::span<const T> Matrix<T>::data() const noexcept {
        return this->data_;
    }

    template<floatTypes T>
    std::size_t Matrix<T>::rows() const noexcept {
        return this->rows_;
    }

    template<floatTypes T>
    std::size_t Matrix<T>::cols() const noexcept {
        return this->cols_;
    }

    template<floatTypes T>
    std::size_t Matrix<T>::stride() const noexcept {
        return stride_;
    }

    template<floatTypes T>
    std::size_t Matrix<T>::bufferSize() const noexcept {
        return rows_ * stride_;
    }

    template<floatTypes T>
    std::size_t Matrix<T>::elementCount() const noexcept {
        return rows_ * cols_;
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::transpose() const {
        Matrix<T> result(cols_, rows_, 0); // Swap rows and col sizes / make sure stride gets recalculated

        for (std::size_t c = 0; c < cols_; ++c) {
            for (std::size_t r = 0; r < rows_; ++r) {
                result(c,r) = (*this)(r,c);
            }
        }

        return result;
    }

    template<floatTypes T>
    T Matrix<T>::mean() const {
        T result = 0;

        for (std::size_t c = 0; c < cols_; ++c) {
            for (std::size_t r = 0; r < rows_; ++r) {
                result += (*this)(r,c);
            }
        }

        return result / (this->rows_ * this->cols_);
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::clip(const T epsilon) const {
        return this->map([&](T num){ return Math::Functions::clamp(num, epsilon);});
    }

    //TODO: Implement matmul more efficiently (e.g row major friendly)
    template<floatTypes T>
    Matrix<T> Matrix<T>::matMul(const Matrix<T> &other) const {
        // this (m x k) other (k x n) (rows x cols) c (m x n)
        if (this->cols_ != other.rows_)
            throw std::invalid_argument("Incompatible matrix sizes");

        Matrix<T> result(this->rows_, other.cols_, 0);

        for (std::size_t r = 0; r < this->rows_; ++r) {
            for (std::size_t c = 0; c < other.cols_; ++c) {
                T sum = T{0};
                for(std::size_t cr = 0; cr < this->cols_; cr++)
                    sum += this->operator()(r,cr) * other(cr, c);
                result(r,c) = sum;
            }
        }
        return result;
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::add(const Matrix<T> &other) const {
        if (this->rows_ != other.rows_ || this->cols_ != other.cols_ || this->stride_ != other.stride_)
            throw std::invalid_argument("In Matrix::add() is not the same size as other (rows/cols/stride)");

        Matrix<T> result(this->rows_, this->cols_, this->stride_);
        for (std::size_t r = 0; r < this->rows_; ++r) {
            for (std::size_t c = 0; c < this->cols_; ++c) {
                result(r,c) = this->operator()(r,c) + other(r,c);
            }
        }
        return result;
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::sub(const Matrix &other) const {
        if (this->rows_ != other.rows_ || this->cols_ != other.cols_ || this->stride_ != other.stride_)
            throw std::invalid_argument("In Matrix::sub() is not the same size as other (rows/cols/stride)");

        Matrix<T> result(this->rows_, this->cols_, this->stride_);
        for (std::size_t r = 0; r < this->rows_; ++r) {
            for (std::size_t c = 0; c < this->cols_; ++c) {
                result(r,c) = this->operator()(r,c) - other(r,c);
            }
        }
        return result;
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::divide(T value) const {
        if(value == 0)
            throw std::invalid_argument("Cant divide by 0 in Matrix divide function");

        return this->map([&](T num){ return num / value;});
    }

    //TODO: Implement swap
    template<floatTypes T>
    void Matrix<T>::swap(const Matrix &other) noexcept {
    }

    template<floatTypes T>
    void Matrix<T>::fill(const T &value) {
        std::fill(this->data_.begin(), this->data_.end(), value);
    }

    template<floatTypes T>
    void Matrix<T>::addInplace(const Matrix &other) {
        if (this->rows_ != other.rows_ || this->cols_ != other.cols_ || this->stride_ != other.stride_)
            throw std::invalid_argument("In Matrix::add() is not the same size as other (rows/cols/stride)");

        for (std::size_t r = 0; r < this->rows_; ++r) {
            for (std::size_t c = 0; c < this->cols_; ++c) {
                this->operator()(r,c) = this->operator()(r,c) + other(r,c);
            }
        }
    }

    template<floatTypes T>
    void Matrix<T>::subInplace(const Matrix &other) {
        if (this->rows_ != other.rows_ || this->cols_ != other.cols_ || this->stride_ != other.stride_)
            throw std::invalid_argument("In Matrix::subInplace() is not the same size as other (rows/cols/stride)");

        for (std::size_t r = 0; r < this->rows_; ++r) {
            for (std::size_t c = 0; c < this->cols_; ++c) {
                this->operator()(r,c) = this->operator()(r,c) - other(r,c);
            }
        }
    }

    // TODO: Improvable by a LOT
    template<floatTypes T>
    Matrix<T> Matrix<T>::hadamard(const Matrix &other) const {
        if(this->rows_ != other.rows_ || this->cols_ != other.cols_)
            throw std::invalid_argument("In Matrix::hadamard() shapes are not the same");

        Matrix<T> result(this->rows_, this->cols_, 0);

        for (std::size_t r = 0; r < this->rows_; ++r) {
            for (std::size_t c = 0; c < this->cols_; ++c) {
                result(r,c) = this->operator()(r,c) * other(r,c);
            }
        }

        return result;
    }

    //TODO: Refer to Todo item for scalar_mul (use two templates, so std:function becomes F&& f)
    template<floatTypes T>
    Matrix<T> Matrix<T>::map(std::function<T(T)> f) const {
        Matrix<T> result(this->rows_, this->cols_, this->stride_);

        for (std::size_t r = 0; r < this->rows_; ++r) {
            for (std::size_t c = 0; c < this->cols_; ++c) {
                result(r,c) = f(this->operator()(r,c));
            }
        }

        return result;
    }

    //TODO: Could be improved by inlining the mapping with return map([alpha](T x){ return alpha * x; }); according to chatgpt
    template<floatTypes T>
    Matrix<T> Matrix<T>::scalarMul(T alpha) const {
        auto temp = [&](T value) { return alpha*value;};
        return map(temp);
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::sumOverColumns() const {
        Matrix<T> result(this->rows_,1, 0);

        for (std::size_t r = 0; r < this->rows_; ++r) {
            T column_sum = {0};
            for (std::size_t c = 0; c < this->cols_; ++c) {
                column_sum += this->operator()(r,c);
            }
            result(r,0) = column_sum;
        }

        return result;
    }

    // TODO: Improvable by a LOT
    template<floatTypes T>
    Matrix<T> Matrix<T>::divide(const Matrix &other) const {
        if(this->rows_ != other.rows_ || this->cols_ != other.cols_ || this->stride_ != other.stride_)
            throw std::invalid_argument("In Matrix::divide() shape or stride are not the same");

        Matrix<T> result(this->rows_, this->cols_, 0);

        for (std::size_t r = 0; r < this->rows_; ++r) {
            for (std::size_t c = 0; c < this->cols_; ++c) {
                result(r,c) = this->operator()(r,c) / other(r,c);
            }
        }

        return result;
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::addBias(const Matrix& bias) const {
        if(bias.cols_ != 1 || bias.rows_ != this->rows_)
            throw std::invalid_argument("In Matrix::addBias either bias matrix has more than 1 column or rows don't match");

        Matrix<T> result(this->rows_, this->cols_, this->stride_);

        for (std::size_t r = 0; r < this->rows_; ++r) {
            for (std::size_t c = 0; c < this->cols_; ++c) {
                result(r,c) = this->operator()(r,c) + bias(r, 0);
            }
        }

        return result;
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::sigmoid() const {
        return this->map([](T num){return Math::Functions::sigmoid(num);});
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::tanh() const {
        return this->map([](T num){return Math::Functions::tanh(num);});
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::relu() const {
        return this->map([](T num){return Math::Functions::relu(num);});
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::softplus() const {
        return this->map([](T num){return Math::Functions::softplus(num);});
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::mish() const {
        return this->map([](T num){return Math::Functions::mish(num);});
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::linear() const {
        return this->map([](T num){return Math::Functions::linear(num);});
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::log(const T base) const {
        return this->map([&](T num){return Math::Functions::log(base, num);});
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::delu(const int a, const int b, const double xc) const {
        return this->map([&](T num){return Math::Functions::delu(num, a, b, xc);});
    }

    template<floatTypes T>
    Matrix<T> Matrix<T>::elu(const double alpha) const {
        return this->map([&](T num){return Math::Functions::elu(num, alpha);});
    }

    template class Matrix<float>;
    template class Matrix<double>;
} // Math