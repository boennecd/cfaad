#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_constructor.hpp>
#include <AAD.h>
#include <vector>
#include <iterator>
#include <algorithm>

using Catch::Approx;
using cfaad::Number;
using std::begin;
using std::end;

namespace {
template<class I1, class I2, class I3>
typename std::iterator_traits<I3>::value_type test_func
(I1 xf, I1 xl, I2 af, I2 al, I3 of, bool trans, size_t const n_ele)
{
    cfaad::matVecProd(xf, xl, af, al, of, trans);
    
    typename std::iterator_traits<I3>::value_type out{0.};
    for(size_t i = 0; i < n_ele; ++i)
        out += sqrt(1 + exp(of[i]));
    
    return out;
}

constexpr size_t m{4}, n{5};
double a[] {-0.47, -0.26, 0.15, 0.82}, 
       B[] {-0.6, 0.8, 0.89, 0.32, 0.26, -0.88, -0.59, -0.65, 0.37, -0.23, 
            0.54, 0, 0.44, 0.98, -0.24, 0.55, 0.87, -0.58, 0.3, -0.75},
       d[] {-0.47, -0.23, -0.97, -0.24, 0.74}, 
      wk[std::max(m,n)];
       

Number ad_a[m],
       ad_B[n * m],
       ad_d[n],
      ad_wk[std::max(m, n)];
       
constexpr double true_val{5.30747316353387},
               true_val_T{6.87572373119088},
                      eps{1e-7};
                      
constexpr double true_derivs[] { -0.222322434773043, -0.102367442765123, -0.110031200131219, -0.0963586241900716, -0.108796085031888, -0.0500947059899589, -0.0538450553167577, -0.0471542203696057, -0.45883566320842, -0.211268977724033, -0.227085668312972, -0.198867798815905, -0.113526349694117, -0.0522727367060539, -0.0561861447727157, -0.0492044038165165, 0.350039578186115, 0.161174271463367, 0.173240612966192, 0.151713578516259, 0.164389406711255, -0.34006602217514, 0.251343908764938, 0.478152596629075, 0.201676156114188 },
               true_derivs_T[] { -0.233168572062211, -0.128986869629047, 0.0744155016863521, 0.406804742695246, -0.111101775058186, -0.0614605563614196, 0.0354580132503882, 0.193837139270273, -0.16208460497828, -0.0896638238127181, 0.0517291292270459, 0.282785906451391, -0.160440990351867, -0.0887545903899025, 0.0512045713977708, 0.279918323700371, -0.0856482647452541, -0.0473798911513097, 0.0273345526905994, 0.149428887408319, 0.20013754925878, 0.338387262824802, 0.461030447186365, 0.0561790116116176 }; 
}

TEST_CASE("cfaad::matVecProd gives the righ value") {    
    /** R code to compute the result
        set.seed(1)
        m <- 4L
        n <- 5L
        dput(a <- round(runif(m, -1, 1), 2))
        dput(B <- round(runif(m * n, -1, 1), 2))
        dput(d <- round(runif(n, -1, 1), 2))

        f <- function(x){
            B <- matrix(head(x, m * n), m)
            d <- tail(x, -m * n)
            sum(sqrt(1 + exp(B %*% d)))
        }

        dput(f(c(B, d)))
        dput(c(numDeriv::grad(f, c(B, d))))

        fT <- function(x){
            B <- matrix(head(x, m * n), m)
            a <- tail(x, -m * n)
            sum(sqrt(1 + exp(crossprod(B, a))))
        }

        dput(fT(c(B, a)))
        dput(c(numDeriv::grad(fT, c(B, a))))
     */
                      
    SECTION("gives the right value with double iterators"){
        double v = test_func(begin(B), end(B), begin(d), end(d), wk, false, m);
        REQUIRE(v == Approx(true_val).epsilon(eps));
        
        v = test_func(begin(B), end(B), begin(a), end(a), wk, true, n);
        REQUIRE(v == Approx(true_val_T).epsilon(eps));
    }
    
    SECTION("gives the right value with a double and a Number iterator"){
        // trans == false
        Number::tape->rewind();
        cfaad::convertCollection(begin(d), end(d), ad_d);
        
        Number v = test_func(begin(B), end(B), begin(ad_d), end(ad_d), ad_wk, 
                             false, m);
        REQUIRE(v.value() == Approx(true_val).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < n; ++i)
            REQUIRE(ad_d[i].adjoint() == 
                Approx(true_derivs[i + n * m]).epsilon(eps)); 
                
        Number::tape->rewind();
        cfaad::convertCollection(begin(B), end(B), ad_B);
        
        v = test_func(begin(ad_B), end(ad_B), begin(d), end(d), ad_wk, false, 
                      m);
        REQUIRE(v.value() == Approx(true_val).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < n * m; ++i)
            REQUIRE(ad_B[i].adjoint() == Approx(true_derivs[i]).epsilon(eps));
        
        // trans == true    
        Number::tape->rewind();
        cfaad::convertCollection(begin(a), end(a), ad_a);
        
        v = test_func(begin(B), end(B), begin(ad_a), end(ad_a), ad_wk, 
                      true, n);
        REQUIRE(v.value() == Approx(true_val_T).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < m; ++i)
            REQUIRE(ad_a[i].adjoint() == 
                Approx(true_derivs_T[i + n * m]).epsilon(eps)); 
                
        Number::tape->rewind();
        cfaad::convertCollection(begin(B), end(B), ad_B);
        
        v = test_func(begin(ad_B), end(ad_B), begin(a), end(a), ad_wk, 
                      true, n);
        REQUIRE(v.value() == Approx(true_val_T).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < n * m; ++i)
            REQUIRE(ad_B[i].adjoint() == 
                Approx(true_derivs_T[i]).epsilon(eps)); 
    }
    
    SECTION("gives the right value with Number iterators"){
        // trans == false
        Number::tape->rewind();
        cfaad::convertCollection(begin(d), end(d), ad_d);
        cfaad::convertCollection(begin(B), end(B), ad_B);
        
        Number v = test_func(begin(ad_B), end(ad_B), begin(ad_d), end(ad_d), 
                             ad_wk, false, m);
        REQUIRE(v.value() == Approx(true_val).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < n * m; ++i)
            REQUIRE(ad_B[i].adjoint() == Approx(true_derivs[i]).epsilon(eps));
        for(size_t i = 0; i < n; ++i)
            REQUIRE(ad_d[i].adjoint() == 
                Approx(true_derivs[i + n * m]).epsilon(eps)); 
                
        // trans == true
        Number::tape->rewind();
        cfaad::convertCollection(begin(a), end(a), ad_a);
        cfaad::convertCollection(begin(B), end(B), ad_B);
        
        v = test_func(begin(ad_B), end(ad_B), begin(ad_a), end(ad_a), ad_wk, 
                      true, n);
        REQUIRE(v.value() == Approx(true_val_T).epsilon(eps));
        v.propagateToStart();
        for(size_t i = 0; i < m; ++i)
            REQUIRE(ad_a[i].adjoint() == 
                Approx(true_derivs_T[i + n * m]).epsilon(eps)); 
        for(size_t i = 0; i < n * m; ++i)
            REQUIRE(ad_B[i].adjoint() == 
                Approx(true_derivs_T[i]).epsilon(eps)); 
    }
}

TEST_CASE("cfaad::matVecProd benchmark"){
    constexpr size_t n_reps{100};
    
    BENCHMARK("two double iterators") {
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(begin(B), end(B), begin(d), end(d), wk, false, m);
        return v;
    };
    
    BENCHMARK("two double iterators (trans)") {
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(begin(B), end(B), begin(a), end(a), wk, true, n);
        return v;
    };
    
    BENCHMARK("mat (double) x vec (Number)") {
        Number::tape->rewind();
        cfaad::convertCollection(begin(a), end(a), ad_a);
        cfaad::convertCollection(begin(d), end(d), ad_d);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(begin(B), end(B), begin(ad_d), end(ad_d), 
                           ad_wk, false, m);
        v.propagateToStart();
        
        return v.value();
    };
    
    BENCHMARK("mat (double) x vec (Number) (trans)") {
        Number::tape->rewind();
        cfaad::convertCollection(begin(a), end(a), ad_a);
        cfaad::convertCollection(begin(d), end(d), ad_d);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(begin(B), end(B), begin(ad_a), end(ad_a), ad_wk, 
                           true, n);
        v.propagateToStart();
        
        return v.value();
    };
    
    BENCHMARK("mat (Number) x vec (double)") {
        Number::tape->rewind();
        cfaad::convertCollection(begin(B), end(B), ad_B);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(begin(ad_B), end(ad_B), begin(d), end(d), 
                           ad_wk, false, m);
        v.propagateToStart();
                       
        return v.value();
    };
    
    BENCHMARK("mat (Number) x vec (double) (trans)") {
        Number::tape->rewind();
        cfaad::convertCollection(begin(B), end(B), ad_B);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(begin(ad_B), end(ad_B), begin(a), end(a), ad_wk, 
                           true, n);
        v.propagateToStart();
                       
        return v.value();
    };
    
    BENCHMARK("two Number iterators") {
        Number::tape->rewind();
        cfaad::convertCollection(begin(a), end(a), ad_a);
        cfaad::convertCollection(begin(d), end(d), ad_d);
        cfaad::convertCollection(begin(B), end(B), ad_B);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(begin(ad_B), end(ad_B), begin(ad_a), end(ad_a), 
                           ad_wk, false, n);
        v.propagateToStart();
                       
        return v.value();
    };
    
    BENCHMARK("two Number iterators (trans)") {
        Number::tape->rewind();
        cfaad::convertCollection(begin(a), end(a), ad_a);
        cfaad::convertCollection(begin(d), end(d), ad_d);
        cfaad::convertCollection(begin(B), end(B), ad_B);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += test_func(begin(ad_B), end(ad_B), begin(ad_a), end(ad_a), 
                           ad_wk, true, n);
        v.propagateToStart();
                       
        return v.value();
    };
}
