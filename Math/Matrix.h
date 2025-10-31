//
// Created by Tim Weissenfels on 10.10.25.
//

#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
#include <span>
#include <type_traits>
#include <ostream>
#include <iomanip>
#include <functional>

/*
 *
 * From ChatGPT:
 * With 2D vectors, rows are separate allocations; memory is not contiguous → poorer spatial locality; matmul is bandwidth-bound and suffers more misses.
 * A flat vector<double> + idx = r*cols + c gives contiguous rows and enables better cache behavior and SIMD friendliness.
 * For didactic clarity, 2D is fine; for performance and scalability, 1D is preferred.
 * Float over double for ML tasks → half the bandwidth, double SIMD width, usually no accuracy issues for XOR/ sin.
 *
 */

namespace Math {

template<typename T>
concept floatTypes = std::is_floating_point_v<T>;

template <floatTypes T>
class Matrix {
private:
    // stride_ >= cols_, data_.size() == rows_*stride_
    std::vector<T> data_;
    std::size_t rows_, cols_, stride_;
public:

    // Con- & Destructors
    explicit Matrix(std::size_t rows, std::size_t cols, std::size_t stride = 0); // if stride=0, round up cols to a SIMD-friendly multiple (e.g., 8 for float on AVX2)
    explicit Matrix() noexcept; // if stride=0, round up cols to a SIMD-friendly multiple (e.g., 8 for float on AVX2)
    Matrix(const Matrix& other) = default;
    Matrix(Matrix&& other) = default;
    ~Matrix() = default;

    // Operators
    Matrix& operator=(const Matrix& other) = default;
    Matrix& operator=(Matrix&& other) = default;
    T& operator()(std::size_t r, std::size_t c);
    const T& operator()(std::size_t r, std::size_t c) const;

    // Getter & Setter
    [[nodiscard]] std::span<const T> data() noexcept;
    [[nodiscard]] std::span<const T> data() const noexcept;
    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t cols() const noexcept;
    [[nodiscard]] std::size_t stride() const noexcept;
    [[nodiscard]] std::size_t bufferSize() const noexcept; // rows_ * stride_
    [[nodiscard]] std::size_t elementCount() const noexcept; // rows_ * cols_

    // Functions
    [[nodiscard]] Matrix transpose() const;
    [[nodiscard]] Matrix matMul(const Matrix& other) const;
    [[nodiscard]] Matrix add(const Matrix& other) const;
    [[nodiscard]] Matrix sub(const Matrix& other) const;
    [[nodiscard]] Matrix divide(T value) const;
    [[nodiscard]] Matrix hadamard(const Matrix& other) const;
    [[nodiscard]] Matrix map(std::function<T(T)> f) const;
    [[nodiscard]] Matrix scalarMul(T value) const;
    [[nodiscard]] Matrix sumOverColumns() const;
    [[nodiscard]] Matrix addBias(const Matrix& bias) const;

    //TODO: Activation functions
    [[nodiscard]] Matrix tanh() const;
    [[nodiscard]] Matrix fastSigmoid_Fabs() const;
    [[nodiscard]] Matrix sigmoid() const;
    [[nodiscard]] Matrix relu() const;
    [[nodiscard]] Matrix elu(const double) const;
    [[nodiscard]] Matrix softplus() const;
    [[nodiscard]] Matrix mish() const;
    [[nodiscard]] Matrix delu(const int a = 1, const int b = 2, const double = 1.25643) const;

    // Inplace functions
    void swap(const Matrix& other) noexcept;
    void fill(const T& value);
    void addInplace(const Matrix& other);
    void subInplace(const Matrix& other);
};

    // ChatGPT generated
    template<floatTypes T>
    std::ostream &operator<<(std::ostream &os, const Matrix<T> &M) {
        // Optional: remember old formatting and restore at end
        std::ios old_state(nullptr);
        old_state.copyfmt(os);

        os << "Matrix(" << M.rows() << "x" << M.cols() << ")\n";
        // choose a cell width; adjust as you like
        constexpr int cell_w = 10;
        os << std::setprecision(6) << std::fixed;

        for (std::size_t r = 0; r < M.rows(); ++r) {
            os << "[ ";
            for (std::size_t c = 0; c < M.cols(); ++c) {
                os << std::setw(cell_w) << M(r, c);
                if (c + 1 < M.cols()) os << ' ';
            }
            os << " ]\n";
        }

        // restore previous formatting
        os.copyfmt(old_state);
        return os;
    }

    extern template class Matrix<float>;
    extern template class Matrix<double>;

} // Math

#endif //MATRIX_H
