//
// Created by timwe on 11/2/2025.
//

// Created by timwe on 11/2/2025.
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "../../Math/Functions.h"

using Catch::Approx;

TEST_CASE("FUNCTIONS") {
    const double EPS = 1e-12;

    SECTION("log base-change correctness") {
        // log base e should match natural log (identity)
        REQUIRE( Math::Functions::log<double,double>(std::exp(1.0), 1.0) == Approx(0.0).margin(1e-15) );
        REQUIRE( Math::Functions::log<double,double>(std::exp(1.0), std::exp(3.0)) == Approx(3.0).epsilon(1e-12) );

        // classic: log_10(100) = 2
        REQUIRE( Math::Functions::log<double,double>(10.0, 100.0) == Approx(2.0).epsilon(1e-12) );

        // float overload sanity
        REQUIRE( Math::Functions::log<float,float>(10.0f, 1000.0f) == Approx(3.0f).epsilon(1e-6f) );
    }

    SECTION("tanh matches std::tanh and is odd") {
        for (double x : { -3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0 }) {
            REQUIRE( tanh(x) == Approx(std::tanh(x)).epsilon(1e-12) );
            REQUIRE( tanh(-x) == Approx(-tanh(x)).epsilon(1e-12) );
        }
    }

    SECTION("sigmoid basic properties") {
        auto sigma = [](double z){ return sigmoid(z); };

        // midpoint and bounds-ish
        REQUIRE( sigma(0.0) == Approx(0.5).epsilon(1e-12) );
        REQUIRE( sigma(8.0)  == Approx(1.0).margin(1e-7) );
        REQUIRE( sigma(-8.0) == Approx(0.0).margin(1e-7) );

        // symmetry: σ(-x) = 1 - σ(x)
        for (double x : {0.1, 0.5, 1.0, 2.0}) {
            REQUIRE( sigma(-x) == Approx(1.0 - sigma(x)).epsilon(1e-12) );
        }
    }

    SECTION("relu piecewise") {
        REQUIRE( Math::Functions::relu(-3.5) == Approx(0.0).margin(0.0) );
        REQUIRE( relu(0.0)  == Approx(0.0).margin(0.0) );
        REQUIRE( relu(2.25) == Approx(2.25).epsilon(1e-12) );

        // float as well
        REQUIRE( relu(-1.0f) == Approx(0.0f).margin(0.0f) );
        REQUIRE( relu(5.0f)  == Approx(5.0f).epsilon(1e-6f) );
    }

    SECTION("softplus equals log(1+exp(x))") {
        auto reference = [](double x){ return std::log1p(std::exp(x)); };
        for (double x : {-20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0}) {
            REQUIRE( softplus(x) == Approx(reference(x)).epsilon(1e-12) );
        }

        // asymptotics (loose checks)
        REQUIRE( softplus(20.0) == Approx(20.0).margin(1e-8) );     // ~ x for large +x
        REQUIRE( softplus(-20.0) == Approx(std::exp(-20.0)).margin(1e-20) ); // ~ e^x for large -x
    }

    SECTION("mish matches reference formula x * tanh(softplus(x))") {
        auto mish_ref = [](double x){
            return x * std::tanh(std::log1p(std::exp(x)));
        };
        for (double x : {-10.0, -3.0, -1.0, 0.0, 0.5, 2.0, 6.0}) {
            REQUIRE( mish(x) == Approx(mish_ref(x)).epsilon(1e-12) );
        }
    }

    SECTION("linear identity") {
        for (double x : {-5.0, -1.2345, 0.0, 3.14159, 10.0}) {
            REQUIRE( linear(x) == Approx(x).epsilon(1e-12) );
        }
        REQUIRE( linear(2.5f) == Approx(2.5f).epsilon(1e-6f) );
    }

    SECTION("elu behavior and parameter validation") {
        // alpha > 0 required (should not throw)
        REQUIRE_NOTHROW( elu(1.0, 1.0) );
        REQUIRE_NOTHROW( elu(-1.0, 0.5) );

        // alpha < 0 should throw
        REQUIRE_THROWS_AS( elu(0.0, -1.0), std::invalid_argument );

        // piecewise: x > 0 => x, else alpha*(e^x-1)
        REQUIRE( elu(2.0, 1.0) == Approx(2.0).epsilon(1e-12) );
        REQUIRE( elu(-1.0, 1.0) == Approx(std::exp(-1.0) - 1.0).epsilon(1e-12) );
        REQUIRE( elu(-1.0, 0.3) == Approx(0.3*(std::exp(-1.0)-1.0)).epsilon(1e-12) );
    }

    SECTION("delu behavior and parameter validation") {
        // According to the docstring, b must be != 0:
        // Expect: b == 0 -> throw; b != 0 -> no throw.
        REQUIRE_THROWS_AS( delu(0.5, /*a*/1, /*b*/0), std::invalid_argument );
        REQUIRE_NOTHROW( delu(0.5, /*a*/1, /*b*/2) );

        // Piecewise with defaults a=1, b=2, xc=1.25643:
        // For x > xc -> x
        REQUIRE( delu(2.0) == Approx(2.0).epsilon(1e-12) );
        // For x <= xc -> (exp(a*x)-1)/b ; at x = 0: (e^0 - 1)/2 = 0
        REQUIRE( delu(0.0) == Approx(0.0).margin(0.0) );
        // A negative input (below xc): (e^{a x} - 1)/b
        REQUIRE( delu(-1.0) == Approx((std::exp(-1.0) - 1.0)/2.0).epsilon(1e-12) );
    }

    SECTION("clip confines to [ε, 1-ε] with default ε=1e-7") {
        const double eps = 1e-7;
        REQUIRE( clip(-1.0) == Approx(eps).margin(0.0) );
        REQUIRE( clip(0.0)  == Approx(eps).margin(0.0) );
        REQUIRE( clip(0.5)  == Approx(0.5).epsilon(1e-12) );
        REQUIRE( clip(1.0)  == Approx(1.0 - eps).margin(0.0) );
        REQUIRE( clip(2.0)  == Approx(1.0 - eps).margin(0.0) );

        // float
        REQUIRE( clip(0.5f) == Approx(0.5f).epsilon(1e-6f) );
    }

    SECTION("clamp confines to [ε, 1-ε] with default ε=1e-7") {
        // Same intent as clip but via std::clamp
        const double eps = 1e-7;
        REQUIRE( clamp(-1.0) == Approx(eps).margin(0.0) );
        REQUIRE( clamp(0.0)  == Approx(eps).margin(0.0) );
        REQUIRE( clamp(0.5)  == Approx(0.5).epsilon(1e-12) );
        REQUIRE( clamp(1.0)  == Approx(1.0 - eps).margin(0.0) );
        REQUIRE( clamp(2.0)  == Approx(1.0 - eps).margin(0.0) );
    }
}
