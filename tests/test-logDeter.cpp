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
template<class V, class I>
V test_func(I Af, I Al, const cfaad::CholFactorization &fact)
{
    auto v = cfaad::logDeter(Af, fact);
    return -2 * v;
}
    
template<class V, class I>
V test_func(I Af, I Al)
{
    const size_t nn{static_cast<size_t>(std::distance(Af, Al))},
                  n{static_cast<size_t>(std::lround(sqrt(nn)))};
    cfaad::CholFactorization fact = cfaad::CholFactorization
        ::getFactorization(Af, static_cast<int>(n));
        
    return test_func<V>(Af, Al, fact);
}

constexpr size_t n{4};
const std::vector<double> A{12.0401617600812, -2.22776598350472, -5.04094088675804, -0.391335183217261, -2.22776598350472, 8.13324744477884, -0.637534321214941, 0.66178316817908, -5.04094088675804, -0.637534321214941, 17.8850549779705, 2.84224374607872, -0.391335183217261, 0.661783168179908, 2.84224374607872, 16.7276837208943};

const cfaad::CholFactorization pre_fact = 
    cfaad::CholFactorization::getFactorization(A.begin(), static_cast<int>(n));

std::vector<Number> ad_A = std::vector<Number>(n * n);
                    
constexpr double true_val{-20.1057752130036},
                      eps{1e-7},
            true_derivs[]{-0.202433862365304, -0.0608365711878335, -0.0604881939798577, 0.00794869866463026, -0.0608365711878335, -0.265968822172829, -0.0288527776765511, 0.0140015097044272, -0.0604881939798577, -0.0288527776765511, -0.133462710621531, 0.0224033814244496, 0.00794869866463026, 0.0140015097044272, 0.0224033814244496, -0.123736870165862};
}

TEST_CASE("cfaad::logDeter gives the right value") {
    /** R code to compute the result
        n <- 4
        set.seed(4)
        A <- drop(rWishart(1, 3 * n, diag(n)))
        dput(c(A))

        f <- function(x)
          -2 * determinant(matrix(x, n))$modulus

        dput(f(c(A)))
        dput(numDeriv::grad(f, c(A)))
        # i.e. 
        dput(-2 * c(solve(A)))
     */
    
    SECTION("double iterator"){
        REQUIRE(test_func<double>(A.begin(), A.end()) 
            == Approx(true_val).epsilon(eps));
    }
    
    SECTION("Number iterator"){
        Number::tape->rewind();
        cfaad::convertCollection(A.begin(), A.end(), ad_A.begin());
        
        Number res = test_func<Number>(ad_A.begin(), ad_A.end());
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < n * n; ++i)
            REQUIRE(ad_A[i].adjoint() == Approx(true_derivs[i]).epsilon(eps));
    }
}

TEST_CASE("cfaad::logDeter benchmark"){
    constexpr size_t n_reps{10};
    
    BENCHMARK("double iterator"){
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<double>(A.begin(), A.end());
            
        return v;
    };
    
    BENCHMARK("double iterator (pre-computed)"){
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<double>(A.begin(), A.end(), pre_fact);
            
        return v;
    };
    
    BENCHMARK("Number iterator"){
        Number::tape->rewind();
        cfaad::convertCollection(A.begin(), A.end(), ad_A.begin());
        
        Number res{0};
        for(size_t i = 0; i < n_reps; ++i)
            res += test_func<Number>(ad_A.begin(), ad_A.end());
            
        res.propagateToStart();
        return res.value();
    };
    
    BENCHMARK("Number iterator  (pre-computed)"){
        Number::tape->rewind();
        cfaad::convertCollection(A.begin(), A.end(), ad_A.begin());
        
        Number res{0};
        for(size_t i = 0; i < n_reps; ++i)
            res += test_func<Number>(ad_A.begin(), ad_A.end(), pre_fact);
            
        res.propagateToStart();
        return res.value();
    };
}

#endif // if AADLAPACK