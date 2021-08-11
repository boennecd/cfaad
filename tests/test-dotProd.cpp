#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_constructor.hpp>
#include <AAD.h>
#include <vector>

using Catch::Approx;
using cfaad::Number;

namespace {
template<class V, class T1, class T2>
V test_func(T1 b1, T1 e1, T2 b2)
{   
    auto v = cfaad::dotProd(b1, e1, b2) - 10;
    return exp(v);
}

const std::vector<double> v1 { 1, 2, 3, 4, 5 },
                          v2 { 2, 1, 2, 1, 0 };
Number n1[5], n2[5];

constexpr double true_val{54.5981500331442}, 
                      eps{1e-10},
            true_val_self{1};
}

TEST_CASE("cfaad::dotProd gives the right value") {
    SECTION("gives the right value with double iterators"){
        REQUIRE(test_func<double>(v1.begin(), v1.end(), v2.begin()) 
            == Approx(true_val).epsilon(eps));
    }
    
    SECTION("gives the right value a double iterator (self)"){
        REQUIRE(test_func<double>(v2.begin(), v2.end(), v2.begin()) 
            == Approx(true_val_self).epsilon(eps));
    }
    
    SECTION("gives the right value with double/Number iterators"){
        Number::tape->rewind();
        cfaad::convertCollection(v1.begin(), v1.end(), n1);
        Number res = test_func<Number>(n1, n1 + v2.size(), v2.begin());
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < v2.size(); ++i)
            REQUIRE(n1[i].adjoint() == Approx(v2[i] * true_val).epsilon(eps));
            
        Number::tape->rewind();
        cfaad::convertCollection(v2.begin(), v2.end(), n2);
        res = test_func<Number>(v1.begin(), v1.end(), n2);
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < v1.size(); ++i)
            REQUIRE(n2[i].adjoint() == Approx(v1[i] * true_val).epsilon(eps));
    }
    
    SECTION("gives the right value with Number/Number iterators"){
        Number::tape->rewind();
        cfaad::convertCollection(v1.begin(), v1.end(), n1);
        cfaad::convertCollection(v2.begin(), v2.end(), n2);
        
        Number res = test_func<Number>(n1, n1 + v1.size(), n2);
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < v2.size(); ++i)
            REQUIRE(n1[i].adjoint() == Approx(v2[i] * true_val).epsilon(eps));
        for(size_t i = 0; i < v1.size(); ++i)
            REQUIRE(n2[i].adjoint() == Approx(v1[i] * true_val).epsilon(eps));
    }
    
    SECTION("gives the right value a Number iterator (self)"){
        Number::tape->rewind();
        cfaad::convertCollection(v2.begin(), v2.end(), n2);
        
        Number res = test_func<Number>(n2, n2 + v2.size(), n2);
        REQUIRE(res.value() == Approx(true_val_self).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < v2.size(); ++i)
            REQUIRE(n2[i].adjoint() 
                == Approx(2 * v2[i] * true_val_self).epsilon(eps));
    }
}

TEST_CASE("cfaad::dotProd benchmark"){
    constexpr size_t n_reps{10};
    
    BENCHMARK("two double iterators"){
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<double>(v1.begin(), v1.end(), v2.begin());
        
        return v;
    };
    
    BENCHMARK("one double iterator (self)"){
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<double>(v2.begin(), v2.end(), v2.begin());
        
        return v;
    };
    
    BENCHMARK("one double and one Number iterator"){
        Number::tape->rewind();
        cfaad::convertCollection(v2.begin(), v2.end(), n2);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<Number>(v1.begin(), v1.end(), n2);
        v.propagateToStart();
        
        return v.value();
    };
    
    BENCHMARK("two Number iterators"){
        Number::tape->rewind();
        cfaad::convertCollection(v1.begin(), v1.end(), n1);
        cfaad::convertCollection(v2.begin(), v2.end(), n2);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<Number>(n1, n1 + v1.size(), n2);
        v.propagateToStart();
        
        return v.value();
    };
    
    BENCHMARK("one Number iterator (self)"){
        Number::tape->rewind();
        cfaad::convertCollection(v2.begin(), v2.end(), n2);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<Number>(n2, n2 + v2.size(), n2);
        v.propagateToStart();
        
        return v.value();
    };
}
