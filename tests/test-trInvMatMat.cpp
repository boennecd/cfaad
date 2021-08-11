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
V test_func(I1 Af, I1 Al, I2 Xf, const cfaad::CholFactorization &fact)
{
    auto v = cfaad::trInvMatMat(Af, Xf, fact);
    return exp(v);
}
    
template<class V, class I1, class I2>
V test_func(I1 Af, I1 Al, I2 Xf)
{
    const size_t nn{static_cast<size_t>(std::distance(Af, Al))},
                  n{static_cast<size_t>(std::lround(sqrt(nn)))};
    cfaad::CholFactorization fact = cfaad::CholFactorization
        ::getFactorization(Af, static_cast<int>(n));
        
    return test_func<V>(Af, Al, Xf, fact);
}

constexpr size_t n{3};
const std::vector<double> A{4.81456966547382, -2.48028473557889, -1.55758896832992, -2.48028473557889, 5.14129687507912, -0.990687733298776, -1.55758896832992, -0.990687733298776, 7.06131384195688},
                          B{1, 2, 3, 2, 4, 5, 3, 5, 6};

const cfaad::CholFactorization pre_fact = 
    cfaad::CholFactorization::getFactorization(A.begin(), static_cast<int>(n));

std::vector<Number> ad_A = std::vector<Number>(n * n),
                    ad_B = std::vector<Number>(n * n);
                    
constexpr double true_val{107.932893119079},
                      eps{1e-7},
            true_derivs[]{-97.6639434153553, -100.555248560455, -73.5480319399573, -100.555248558463, -103.888593707723, -75.0523732759246, -73.5480319405439, -75.0523732867015, -53.4731020097184, 35.7992236776067, 19.3141639942999, 10.6063521452268, 19.3141639942999, 31.9968818874742, 8.74942659985218, 10.6063521455412, 8.74942660004674, 18.8521828654125};
}

TEST_CASE("cfaad::trInvMatMat gives the right value") {
    /** R code to compute the result
        n <- 3
        set.seed(2)
        A <- drop(rWishart(1, 3 * n, diag(n)))
        B <- matrix(0., n, n)
        B[lower.tri(B, TRUE)] <- 1:((n * (n + 1))/2)
        B[upper.tri(B)] <- t(B)[upper.tri(B)]
        dput(c(A))
        dput(c(B))

        f <- function(x){
          X1 <- matrix(head(x, n * n), n)
          X2 <- matrix(tail(x, -n * n), n)
          exp(sum(diag(solve(X1, X2))))
        }

        dput(f(c(A, B)))
        dput(numDeriv::grad(f, c(A, B)))
     */
    
    SECTION("two double iterators"){
        REQUIRE(test_func<double>(A.begin(), A.end(), B.begin()) 
            == Approx(true_val).epsilon(eps));
    }
    
    SECTION("Numbers and doubles"){
        Number::tape->rewind();
        cfaad::convertCollection(A.begin(), A.end(), ad_A.begin());
        
        Number res = test_func<Number>(ad_A.begin(), ad_A.end(), B.begin());
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < n * n; ++i)
            REQUIRE(ad_A[i].adjoint() == Approx(true_derivs[i]).epsilon(eps));
    }
    
    SECTION("doubles and Numbers"){
        Number::tape->rewind();
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number res = test_func<Number>(A.begin(), A.end(), ad_B.begin());
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < n * n; ++i)
            REQUIRE(ad_B[i].adjoint() 
                == Approx(true_derivs[i + n * n]).epsilon(eps));
    }
    
    SECTION("Numbers and Numbers"){
        Number::tape->rewind();
        cfaad::convertCollection(A.begin(), A.end(), ad_A.begin());
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number res = test_func<Number>(ad_A.begin(), ad_A.end(), ad_B.begin());
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < n * n; ++i)
            REQUIRE(ad_A[i].adjoint() == Approx(true_derivs[i]).epsilon(eps));
        for(size_t i = 0; i < n * n; ++i)
            REQUIRE(ad_B[i].adjoint() 
                == Approx(true_derivs[i + n * n]).epsilon(eps));
    }
}

TEST_CASE("cfaad::trInvMatMat benchmark"){
    constexpr size_t n_reps{10};
    
    BENCHMARK("two double iterators"){
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<double>(A.begin(), A.end(), B.begin());
            
        return v;
    };
    
    BENCHMARK("two double iterators (pre-computed)"){
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func<double>(A.begin(), A.end(), B.begin(), pre_fact);
            
        return v;
    };
    
    BENCHMARK("Numbers and doubles"){
        Number::tape->rewind();
        cfaad::convertCollection(A.begin(), A.end(), ad_A.begin());
        
        Number res{0};
        for(size_t i = 0; i < n_reps; ++i)
            res += test_func<Number>(ad_A.begin(), ad_A.end(), B.begin());
            
        res.propagateToStart();
        return res.value();
    };
    
    BENCHMARK("doubles and Numbers"){
        Number::tape->rewind();
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number res{0};
        for(size_t i = 0; i < n_reps; ++i)
            res += test_func<Number>(A.begin(), A.end(), ad_B.begin());
            
        res.propagateToStart();
        return res.value();
    };
    
    BENCHMARK("Numbers and Numbers"){
        Number::tape->rewind();
        cfaad::convertCollection(A.begin(), A.end(), ad_A.begin());
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number res{0};
        for(size_t i = 0; i < n_reps; ++i)
            res += test_func<Number>(ad_A.begin(), ad_A.end(), ad_B.begin());
            
        res.propagateToStart();
        return res.value();
    };
    
    BENCHMARK("Numbers and Numbers (pre-computed)"){
        Number::tape->rewind();
        cfaad::convertCollection(A.begin(), A.end(), ad_A.begin());
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number res{0};
        for(size_t i = 0; i < n_reps; ++i)
            res += test_func<Number>(ad_A.begin(), ad_A.end(), ad_B.begin(),
                                     pre_fact);
            
        res.propagateToStart();
        return res.value();
    };
}

#endif // if AADLAPACK