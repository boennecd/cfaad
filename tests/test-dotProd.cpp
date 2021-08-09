#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
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
}

TEST_CASE("cfaad::dotProd gives the righ value") {
    std::vector<double> v1 { 1, 2, 3, 4, 5 },
                        v2 { 2, 1, 2, 1, 0 };
    Number n1[5], n2[5];
    
    constexpr double true_val{54.5981500331442}, 
                          eps{1e-10};
                      
    SECTION("gives the right value with double iterators"){
        REQUIRE(test_func<double>(v1.begin(), v1.end(), v2.begin()) 
            == Approx(true_val).epsilon(eps));
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
        
        res.propagateToStart();
        for(size_t i = 0; i < v2.size(); ++i)
            REQUIRE(n1[i].adjoint() == Approx(v2[i] * true_val).epsilon(eps));
        for(size_t i = 0; i < v1.size(); ++i)
            REQUIRE(n2[i].adjoint() == Approx(v1[i] * true_val).epsilon(eps));
    }
}

