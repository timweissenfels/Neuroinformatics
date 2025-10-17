//
// Created by Tim Weissenfels on 10.10.25.
//

#include "Matrix.h"

#include <algorithm>
//#include <arm_neon.h>
#include <iostream>

namespace Math {

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
    std::span<const T> Matrix<T>::data() noexcept {
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
    Matrix<T> Matrix<T>::matmul(const Matrix &other) const {
        if (this->cols_ != other.rows_)
            throw std::invalid_argument("Incompatible matrix sizes");

        Matrix result(this->rows_, other.cols_, 0);

        for (std::size_t r = 0; r < other.rows_; ++r) {
            for (std::size_t c = 0; c < other.cols_; ++c) {

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
            throw std::invalid_argument("In Matrix::add() is not the same size as other (rows/cols/stride)");

        Matrix<T> result(this->rows_, this->cols_, this->stride_);
        for (std::size_t r = 0; r < this->rows_; ++r) {
            for (std::size_t c = 0; c < this->cols_; ++c) {
                result(r,c) = this->operator()(r,c) - other(r,c);
            }
        }
        return result;
    }

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
            throw std::invalid_argument("In Matrix::add() is not the same size as other (rows/cols/stride)");

        for (std::size_t r = 0; r < this->rows_; ++r) {
            for (std::size_t c = 0; c < this->cols_; ++c) {
                this->operator()(r,c) = this->operator()(r,c) - other(r,c);
            }
        }
    }

    template class Matrix<float>;
    template class Matrix<double>;
} // Math