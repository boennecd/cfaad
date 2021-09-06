#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_constructor.hpp>
#include <AAD.h>
#include <array>
#include <iterator>

using Catch::Approx;
using cfaad::Number;
using std::begin;
using std::end;


/* R code to reproduce the result
    n <- 4
    set.seed(101)
    B <- drop(rWishart(1, 3 * n, diag(n)))
    b <- rnorm(n)
    dput(b)
    dput(c(B))

    f <- function(x){
      X <- matrix(head(x, n * n), n)
      x <- tail(x, -n * n)
      x %*% X %*% x
    }

    dput(f(c(B, b)))
    dput(numDeriv::grad(f, c(B, b))) # though easy to compute...
*/

namespace {
constexpr size_t n{4};
constexpr double X[]{ 9.52390320318291, -1.61785642904986, 3.62295643644341, -2.34035371924878, -1.61785642904986, 12.1737549450385, 1.51906092051661, -5.69781716551355, 3.62295643644341, 1.51906092051661, 9.02784238593205, 2.63969054950944, -2.34035371924878, -5.69781716551355, 2.63969054950944, 14.1945644507719 },
                 y[]{0.415361897650897, 0.826429519891201, -0.280862936968038, 0.410703656608391};
                 
constexpr double true_val{5.127515045596},
                      eps{1e-7},
              true_derivs[]{0.17252550601816, 0.343267333667258, -0.116659762468064, 0.170590650187647, 0.343267333693617, 0.682985751354937, -0.23211342218183, 0.339417625759409, -0.11665976247998, -0.232113422181656, 0.0788839893569425, -0.115351435213872, 0.17059065018405, 0.339417625751925, -0.115351435230127, 0.168677493551263,  1.2801523601279, 13.2439845638, 2.61755816392777, -1.18513956618962};

Number ad_X[n * n], 
       ad_y[n];

} // namespace

TEST_CASE("cfaad::quadFormSym gives the right value") {    
    SECTION("two double iterators"){
        REQUIRE(cfaad::quadFormSym(X, y, end(y)) 
            == Approx(true_val).epsilon(eps));
    }
    
    SECTION("mat (double) x vec (Number)") {
        Number::tape->rewind();
        cfaad::convertCollection(y, end(y), ad_y);
        
        Number res = cfaad::quadFormSym(X, ad_y, end(ad_y));
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < n; ++i)
            REQUIRE(ad_y[i].adjoint() == 
                Approx(true_derivs[i + n * n]).epsilon(eps));
    }
    
    SECTION("mat (Number) x vec (double)") {
        Number::tape->rewind();
        cfaad::convertCollection(X, end(X), ad_X);
        
        Number res = cfaad::quadFormSym(ad_X, y, end(y));
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < n * n; ++i)
            REQUIRE(ad_X[i].adjoint() == Approx(true_derivs[i]).epsilon(eps));
    }
    
    SECTION("mat (Number) x vec (Number)") {
        Number::tape->rewind();
        cfaad::convertCollection(X, end(X), ad_X);
        cfaad::convertCollection(y, end(y), ad_y);
        
        Number res = cfaad::quadFormSym(ad_X, ad_y, end(ad_y));
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        
        res.propagateToStart();
        for(size_t i = 0; i < n * n; ++i)
            REQUIRE(ad_X[i].adjoint() == Approx(true_derivs[i]).epsilon(eps));
        for(size_t i = 0; i < n; ++i)
            REQUIRE(ad_y[i].adjoint() == 
                Approx(true_derivs[i + n * n]).epsilon(eps));
    }
}

TEST_CASE("cfaad::quadFormSym benchmark (10x y^TXy with a 4x4 matrix)"){
    constexpr size_t n_reps{10};
    
    BENCHMARK("two double iterators"){
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += cfaad::quadFormSym(X, y, end(y));
            
        return v;
    };
    
    
    BENCHMARK("mat (double) x vec (Number)"){
        Number::tape->rewind();
        cfaad::convertCollection(y, end(y), ad_y);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += cfaad::quadFormSym(X, ad_y, end(ad_y));
            
        v.propagateToStart();
        return v.value();
    };
    
    BENCHMARK("mat (Number) x vec (double)"){
        Number::tape->rewind();
        cfaad::convertCollection(X, end(X), ad_X);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += cfaad::quadFormSym(ad_X, y, end(y));
            
        v.propagateToStart();
        return v.value();
    };
    
    BENCHMARK("mat (Number) x vec (Number)"){
        Number::tape->rewind();
        cfaad::convertCollection(X, end(X), ad_X);
        cfaad::convertCollection(y, end(y), ad_y);
        
        Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += cfaad::quadFormSym(ad_X, ad_y, end(ad_y));
            
        v.propagateToStart();
        return v.value();
    };
}