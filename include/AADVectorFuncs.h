#pragma once

#include "AADNumWrapper.h"
#include <numeric>

namespace cfaad {
// sum function

template<class I, class V>
struct VecSumOp {
    /// general case
    static V sum(I begin, I end){
        return std::accumulate(begin, end, V{0.});
    }
};

template<class I>
struct VecSumOp<I, Number> {
    /// sum over range T iterator
    static Number sum(I begin, I end){
        return Number::sum(begin, end);
    }
};

template<class I>
it_value_type<I> sum(I begin, I end){
    return VecSumOp<I, it_value_type<I> >(begin, end);
}

// dot product

template<class I1, class I2, class V1, class V2>
struct VecDotProdOp {
  using returnT = V1;
    /// general case
    static V1 dot_prodcut
    (I1 first1, I1 last1, I2 first2){
        return std::inner_product(first1, last1, first2, V1{0.});
    }
};

template<class I1, class I2, class V2>
struct VecDotProdOp<I1, I2, Number, V2> {
    using returnT = Number;
    /// dot product with one T and none T iterator
    static Number dot_prodcut
    (I1 first1, I1 last1, I2 first2){
        return Number::dot_product(first1, last1, first2);
    }
};

template<class I1, class I2, class V1>
struct VecDotProdOp<I1, I2, V1, Number> {
    using returnT = Number;
    /// dot product with one T and none T iterator
    static Number dot_prodcut
    (I1 first1, I1 last1, I2 first2){
        auto const n = std::distance(first1, last1);
        return Number::dot_product(first2, first2 + n, first1);
    }
};

template<class I1, class I2>
struct VecDotProdOp<I1, I2, Number, Number> {
    using returnT = Number;
    /// dot product with two T iterators
    static Number dot_prodcut
        (I1 first1, I1 last1, I2 first2){
        return Number::dot_product_identical(first1, last1, first2);
    }
};

template<class I1, class I2>
typename VecDotProdOp<I1, I2, it_value_type<I1>, it_value_type<I2> >::returnT
dot_prod(I1 first1, I1 last1, I2 first2){
  return VecDotProdOp
      <I1, I2, it_value_type<I1>, it_value_type<I2> >::dot_prodcut
      (first1, last1, first2);
}

} // namespace cfadd
