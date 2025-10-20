#include <iostream>
#include "Math/Matrix.h"
// -O3 -DNDEBUG for speed; -DML_DEBUG to enable shape asserts. -mcpu=apple-m1 -ffast-math

bool test_hadamard_product() {
    auto A = Math::Matrix<double>(3, 2);
    auto B = Math::Matrix<double>(3, 2);
    auto C = Math::Matrix<double>(3, 2);

    A(0, 0) = 2; A(0,1) = 4;
    A(1, 0) = 1; A(1,1) = 3;
    A(2, 0) = 5; A(2,1) = 2;

    B(0, 0) = 3; B(0,1) = 1;
    B(1, 0) = 2; B(1,1) = 4;
    B(2, 0) = 1; B(2,1) = 6;

    C(0, 0) = 6; C(0,1) = 4;
    C(1, 0) = 2; C(1,1) = 12;
    C(2, 0) = 5; C(2,1) = 12;

   auto C_check = A.hadamard(B);

   if(std::equal(std::begin(C_check.data()), std::end(C_check.data()), std::begin(C.data()), std::end(C.data())))
        return true;
   else
       return false;
}

int main() {
    auto A = Math::Matrix<double>(3, 2);
    auto B = Math::Matrix<double>(3, 2);

    A.fill(5);
    B.fill(9);

    std::cout << A << std::endl;
    std::cout << B << std::endl;

    auto result = A.add(B);

    std::cout << result << std::endl;

    result = result.sub(B);

    std::cout << result << std::endl;

    auto transpose = result.transpose();

    std::cout << transpose << std::endl;

    std::cout << transpose.scalarMul(5) << std::endl;

    auto cubed = [](auto x) { return x*x*x;};

    std::cout << transpose.map(cubed) << std::endl;

    std::cout << transpose.sumOverColumns() << std::endl;

    auto bias = Math::Matrix<double>(2, 1, 0);
    bias(0,0) = 1; bias(1,0) = 2;

    std::cout << transpose.addBias(bias) << std::endl;

    if(test_hadamard_product())
        std::cout << "Hadamard check works" << std::endl;
    else
        std::cout << "Hadamard does not works" << std::endl;

    auto mulA = Math::Matrix<double>(3, 2);
    auto mulB = Math::Matrix<double>(2, 4);

    mulA(0,0) = 1; mulA(0,1) = 2;
    mulA(1,0) = 9; mulA(1,1) = 3;
    mulA(2,0) = 7; mulA(2,1) = 4;

    mulB(0,0) = 1; mulB(0,1) = 6; mulB(0,2) = 3; mulB(0,3) = 8;
    mulB(1,0) = 4; mulB(1,1) = 9; mulB(1,2) = 2; mulB(1,3) = 3;

    std::cout << mulA << std::endl;
    std::cout << mulB << std::endl;

    std::cout << mulA.matMul(mulB) << std::endl;

    return 0;
}