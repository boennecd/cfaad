#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <AAD.h>
#include <list>

#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/benchmark/catch_constructor.hpp>

using Catch::Approx;

namespace {
template<class T>
typename std::iterator_traits<T>::value_type log_sum(T begin, T end)
{   
    return log(cfaad::sum(begin, end));
}

std::list<double> dat { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    cfaad::Number num_dat[10];
constexpr double true_val{4.00733318523247}, 
                      eps{1e-10}, 
                 true_der{1./55.};
}

TEST_CASE("cfaad::sum gives the right value") {
    SECTION("gives the right value with double iterators"){
        REQUIRE(
            log_sum(dat.begin(), dat.end()) == Approx(true_val).epsilon(eps));
    }
    
    SECTION("gives the right value with Number iterators"){
        cfaad::Number::tape->rewind();
        cfaad::convertCollection(dat.begin(), dat.end(), num_dat);
        cfaad::Number res = log_sum(num_dat, num_dat + dat.size());
        REQUIRE(res.value() == Approx(true_val).epsilon(eps));
        res.propagateToStart();
        for(size_t i = 0; i < dat.size(); ++i)
            REQUIRE(num_dat[i].adjoint() == Approx(true_der).epsilon(eps));
    }
}

TEST_CASE("cfaad::sum benchmark (10x with a 10D vector)") {
    constexpr size_t n_reps{10};
    
    BENCHMARK("double iterator") {
        double v{};
        for(size_t i = 0; i < n_reps; ++i)
            v += log_sum(dat.begin(), dat.end());
        return v;
    };
    
    BENCHMARK("Number iterator") {
        cfaad::Number::tape->rewind();
        cfaad::convertCollection(dat.begin(), dat.end(), num_dat);
        cfaad::Number v{0};
        for(size_t i = 0; i < n_reps; ++i)
            v += log_sum(num_dat, num_dat + dat.size());
        v.propagateToStart();
        return v.value();
    };
}