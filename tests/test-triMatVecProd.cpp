#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_constructor.hpp>
#include <AAD.h>
#include <array>

using Catch::Approx;
using cfaad::Number;

namespace {
template<class I1, class I2, class I3>
typename std::iterator_traits<I3>::value_type test_func
(I1 xf, I2 af, I2 al, I3 of, bool trans)
{
    const size_t n_ele{static_cast<size_t>(std::distance(af, al))};
    cfaad::triMatVecProd(xf, af, al, of, trans);
    return cfaad::dotProd(of, of + n_ele, of);
}

constexpr size_t n{5}, 
         n_mat_ele{(n * (n + 1)) / 2};
const std::array<double, n> a {-1, 1, -2, 2, -3};
const std::array<double, n_mat_ele> B { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
std::array<double, n> wk;

std::array<Number, n> ad_a, ad_wk;
std::array<Number, n_mat_ele> ad_B;

const double true_val{5003}, 
           true_val_T{1896}, 
                  eps{1e-8}, 
        true_derivs[] { 52, -52, -54, 104, 108, 132, -104, -108, -132, -88, 156, 162, 198, 132, 270, -52, -266, -874, -1830, -4044 },
      true_derivs_T[] { 2, -2, 2, 22, -22, 44, -6, 6, -12, 12, 84, -84, 168, -168, 252, -968, -1064, -1170, -1116, -1260 };
}

TEST_CASE("cfaad::triMatVecProd gives the right value") {    
    /** R code to compute the result
        n <- 5L
        b <- 1:15
        a <- c(-1, 1, -2, 2, -3)

        f <- function(x){
            B <- matrix(0, n, n)
            B[upper.tri(B, TRUE)] <- head(x, n * (n + 1) / 2)
            a <- tail(x, -n * (n + 1) / 2)
            sum((B %*% a)^2)
        }

        dput(f(c(b, a)))
        dput(round(c(numDeriv::grad(f, c(b, a))), 7))

        fT <- function(x){
            B <- matrix(0, n, n)
            B[upper.tri(B, TRUE)] <- head(x, n * (n + 1) / 2)
            a <- tail(x, -n * (n + 1) / 2)
            sum(crossprod(B, a)^2)
        }

        dput(fT(c(b, a)))
        dput(round(c(numDeriv::grad(fT, c(b, a))), 7))
     */
                      
    SECTION("gives the right value with double iterators"){
        double v = test_func(B.begin(), a.begin(), a.end(), wk.begin(), false);
        REQUIRE(v == Approx(true_val).epsilon(eps));
        
        v = test_func(B.begin(), a.begin(), a.end(), wk.begin(), true);
        REQUIRE(v == Approx(true_val_T).epsilon(eps));
    }
    
    SECTION("gives the right value with a double and a Number iterator"){
        // trans == false
        Number::tape->rewind();
        cfaad::convertCollection(a.begin(), a.end(), ad_a.begin());
        
        Number v = test_func(B.begin(), ad_a.begin(), ad_a.end(), ad_wk.begin(), 
                             false);
        REQUIRE(v.value() == Approx(true_val).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < n; ++i)
            REQUIRE(ad_a[i].adjoint() == 
                Approx(true_derivs[i + n_mat_ele]).epsilon(eps)); 
                
        Number::tape->rewind();
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        v = test_func(ad_B.begin(), a.begin(), a.end(), ad_wk.begin(), 
                      false);
        REQUIRE(v.value() == Approx(true_val).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < n_mat_ele; ++i)
            REQUIRE(ad_B[i].adjoint() == Approx(true_derivs[i]).epsilon(eps)); 
            
        // trans == true
        Number::tape->rewind();
        cfaad::putOnTape(ad_a.begin(), ad_a.end());
        
        v = test_func(B.begin(), ad_a.begin(), ad_a.end(), ad_wk.begin(), 
                      true);
        REQUIRE(v.value() == Approx(true_val_T).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < n; ++i)
            REQUIRE(ad_a[i].adjoint() == 
                Approx(true_derivs_T[i + n_mat_ele]).epsilon(eps)); 
                
        Number::tape->rewind();
        cfaad::putOnTape(ad_B.begin(), ad_B.end());
        
        v = test_func(ad_B.begin(), a.begin(), a.end(), ad_wk.begin(), 
                      true);
        REQUIRE(v.value() == Approx(true_val_T).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < n_mat_ele; ++i)
            REQUIRE(ad_B[i].adjoint() == Approx(true_derivs_T[i]).epsilon(eps)); 
    }
    
    SECTION("gives the right value with Number iterators"){
        // trans == false
        Number::tape->rewind();
        cfaad::convertCollection(a.begin(), a.end(), ad_a.begin());
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number v = test_func(ad_B.begin(), ad_a.begin(), ad_a.end(), 
                             ad_wk.begin(), false);
        REQUIRE(v.value() == Approx(true_val).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < n_mat_ele; ++i)
            REQUIRE(ad_B[i].adjoint() == Approx(true_derivs[i]).epsilon(eps)); 
        for(size_t i = 0; i < n; ++i)
            REQUIRE(ad_a[i].adjoint() == 
                Approx(true_derivs[i + n_mat_ele]).epsilon(eps)); 
        
        // trans == true
        Number::tape->rewind();
        cfaad::putOnTape(ad_a.begin(), ad_a.end());
        cfaad::putOnTape(ad_B.begin(), ad_B.end());
        
        v = test_func(ad_B.begin(), ad_a.begin(), ad_a.end(), ad_wk.begin(), 
                      true);
        REQUIRE(v.value() == Approx(true_val_T).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < n_mat_ele; ++i)
            REQUIRE(ad_B[i].adjoint() == Approx(true_derivs_T[i]).epsilon(eps));
        for(size_t i = 0; i < n; ++i)
            REQUIRE(ad_a[i].adjoint() == 
                Approx(true_derivs_T[i + n_mat_ele]).epsilon(eps));
    }
}

TEST_CASE("cfaad::triMatVecProd benchmark"){
    constexpr size_t n_reps{100};
    
    BENCHMARK("two double iterators") {
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(B.begin(), a.begin(), a.end(), wk.begin(), false);
        return v;
    };
    
    BENCHMARK("two double iterators (trans)") {
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(B.begin(), a.begin(), a.end(), wk.begin(), true);
        return v;
    };
    
    BENCHMARK("mat (double) x vec (Number)") {
        Number::tape->rewind();
        cfaad::convertCollection(a.begin(), a.end(), ad_a.begin());
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(B.begin(), ad_a.begin(), ad_a.end(), ad_wk.begin(), 
                           false);
        v.propagateToStart();
        
        return v.value();
    };
    
    BENCHMARK("mat (double) x vec (Number) (trans)") {
        Number::tape->rewind();
        cfaad::convertCollection(a.begin(), a.end(), ad_a.begin());
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(B.begin(), ad_a.begin(), ad_a.end(), ad_wk.begin(), 
                           true);
        v.propagateToStart();
        
        return v.value();
    };
    
    BENCHMARK("mat (Number) x vec (double)") {
        Number::tape->rewind();
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(ad_B.begin(), a.begin(), a.end(), ad_wk.begin(), 
                           false);
        v.propagateToStart();
                       
        return v.value();
    };
    
    BENCHMARK("mat (Number) x vec (double) (trans)") {
        Number::tape->rewind();
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(ad_B.begin(), a.begin(), a.end(), ad_wk.begin(), 
                           true);
        v.propagateToStart();
                       
        return v.value();
    };
    
    BENCHMARK("two Number iterators") {
        Number::tape->rewind();
        cfaad::convertCollection(a.begin(), a.end(), ad_a.begin());
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(ad_B.begin(), ad_a.begin(), ad_a.end(), 
                           ad_wk.begin(), false);
        v.propagateToStart();
                       
        return v.value();
    };
    
    BENCHMARK("two Number iterators (trans)") {
        Number::tape->rewind();
        cfaad::convertCollection(a.begin(), a.end(), ad_a.begin());
        cfaad::convertCollection(B.begin(), B.end(), ad_B.begin());
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(ad_B.begin(), ad_a.begin(), ad_a.end(), 
                           ad_wk.begin(), true);
        v.propagateToStart();
                       
        return v.value();
    };
}