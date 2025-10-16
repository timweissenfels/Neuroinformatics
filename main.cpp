#include <iostream>
#include "Math/Matrix.h"
// -O3 -DNDEBUG for speed; -DML_DEBUG to enable shape asserts. -mcpu=apple-m1 -ffast-math

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

    return 0;
}