#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_constructor.hpp>
#include <AAD.h>
#include <vector>

using Catch::Approx;
using cfaad::Number;

#if AADLAPACK

namespace {
// version where the factorization is computed
template<class V, class I1, class I2>
V test_func(I1 af, I1 al, I2 xf, const cfaad::CholFactorization &fact)
{
    auto v = cfaad::quadFormInv(af, xf, fact);
    return 3 * v;
}
    
template<class V, class I1, class I2>
V test_func(I1 af, I1 al, I2 xf)
{
    const size_t n{static_cast<size_t>(std::distance(af, al))};
    cfaad::CholFactorization fact = cfaad::CholFactorization
        ::getFactorization(xf, static_cast<int>(n), false);
        
    return test_func<V>(af, al, xf, fact);
}

constexpr size_t n{3};
const std::vector<double> a{-0.29472044679056, -0.00576717274753696, 2.40465338885795},
                          B{5.69040694552746, 3.03532592432841, -3.67348520317816, 3.03532592432841, 14.4789126529084, -5.28937597574071, -3.67348520317816, -5.28937597574071, 10.7560066919034};

const cfaad::CholFactorization pre_fact = 
    cfaad::CholFactorization::getFactorization(B.begin(), static_cast<int>(n));

std::vector<Number> ad_a = std::vector<Number>(n),
                    ad_B = std::vector<Number>(n * n);
                    
constexpr double true_val{2.07661372009289},
                      eps{1e-7},
            true_derivs[]{-0.0264790861576653, -0.0251832965786533, -0.0844381011777485, -0.0251832965857209, -0.0239509182292566, -0.0803060095502879, -0.0844381011858454, -0.0803060095454358, -0.269261291305596, 0.563692321875334, 0.536107283090719, 1.79753595114407};
}

TEST_CASE("cfaad::quadFormInv gives the right value") {
    /** R code to compute the result
        n <- 3
        set.seed(1)
        B <- drop(rWishart(1, 3 * n, diag(n)))
        b <- rnorm(n)
        dput(b)
        dput(c(B))

        f <- function(x){
          X <- matrix(head(x, n * n), n)
          x <- tail(x, -n * n)
          v <- solve(X, x)
          3 * sum(x %*% v)
        }

        dput(f(c(B, b)))
        dput(numDeriv::grad(f, c(B, b)))
     */
    
    SECTION("two double iterators"){
        REQUIRE(test_func<double>(a.begin(), a.end(), B.begin()) 
            == Approx(true_val).epsilon(eps));
    }
    
    SECTION("mat (double) x vec (Number)") {
        Number::tape->rewind();
        cfaad::convertCollection(a.begin(), a.end(), ad_a.begin());
        
        Number res = test_func<Number>(ad_a.begin(), ad_a.end(), B.begin());
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < n; ++i)
            REQUIRE(ad_a[i].adjoint() 
                == Approx(true_derivs[i + n * n]).epsilon(eps));
    }
    
    SECTION("mat (Number) x vec (double)") {
        Number::tape->rewind();
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number res = test_func<Number>(a.begin(), a.end(), ad_B.begin());
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < n * n; ++i)
            REQUIRE(ad_B[i].adjoint() == Approx(true_derivs[i]).epsilon(eps));
    }
    
    SECTION("two Number iterators") {
        Number::tape->rewind();
        cfaad::convertCollection(a.begin(), a.end(), ad_a.begin());
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number res = test_func<Number>(ad_a.begin(), ad_a.end(), ad_B.begin());
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < n; ++i)
            REQUIRE(ad_a[i].adjoint() 
                == Approx(true_derivs[i + n * n]).epsilon(eps));
        for(size_t i = 0; i < n * n; ++i)
            REQUIRE(ad_B[i].adjoint() == Approx(true_derivs[i]).epsilon(eps));
    }
}

TEST_CASE("cfaad::quadFormInv benchmark (10x aB^{-1}a with a 3x3 matrix)"){
    constexpr size_t n_reps{10};
    
    BENCHMARK("two double iterators"){
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<double>(a.begin(), a.end(), B.begin());
            
        return v;
    };
    
    BENCHMARK("two double iterators (pre-computed factorization)"){
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<double>(a.begin(), a.end(), B.begin(), pre_fact);
            
        return v;
    };
    
    BENCHMARK("mat (double) x vec (Number)"){
        Number::tape->rewind();
        cfaad::convertCollection(a.begin(), a.end(), ad_a.begin());
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<Number>(ad_a.begin(), ad_a.end(), B.begin());
            
        v.propagateToStart();
        return v.value();
    };
    
    BENCHMARK("mat (Number) x vec (double)"){
        Number::tape->rewind();
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<Number>(a.begin(), a.end(), ad_B.begin());
            
        v.propagateToStart();
        return v.value();
    };
    
    BENCHMARK("two Number iterators"){
        Number::tape->rewind();
        cfaad::convertCollection(a.begin(), a.end(), ad_a.begin());
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<Number>(ad_a.begin(), ad_a.end(), ad_B.begin());
            
        v.propagateToStart();
        return v.value();
    };
    
    BENCHMARK("two Number iterators (pre-computed factorization)"){
        Number::tape->rewind();
        cfaad::convertCollection(a.begin(), a.end(), ad_a.begin());
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<Number>(ad_a.begin(), ad_a.end(), ad_B.begin(), 
                                   pre_fact);
            
        v.propagateToStart();
        return v.value();
    };
}

#endif // if AADLAPACK