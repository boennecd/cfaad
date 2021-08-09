#pragma once

#include <iterator>
#include <type_traits>

namespace cfaad {
template<class T>
using it_value_type = typename std::iterator_traits<T>::value_type;

template<class T, class U>
using is_it_value_type = std::is_same<it_value_type<T>, U>;

template<class T>
struct vectorOps {
    /// computes the sum
    template<class I>
    static T sum(I begin, I end){
        static_assert(is_it_value_type<I, T>::value, "Iterator not Ts");

        T res;
        res.createNode(static_cast<size_t>(std::distance(begin, end)));

        double val{};
        size_t i{};
        for(; begin != end; ++begin, ++i){
            val += begin->value();
            res.setpDerivatives(i, 1.);
            res.setpAdjPtrs(i, *begin);
        }

        res.myValue = val;
        return res;
    }

    /// computes the dot product with T and none T iterator
    template<class I1, class I2>
    static T dot_product(I1 f1, I1 l1, I2 f2){
        static_assert(is_it_value_type<I1, T>::value,
                      "First iterator not to T");
        static_assert(!is_it_value_type<I2, T>::value,
                      "Second iterator to T");

        T res;
        res.createNode(static_cast<size_t>(std::distance(f1, l1)));

        double val{};
        size_t i{};
        for(; f1 != l1; ++f1, ++f2, ++i){
          val += f1->value() * *f2;
          res.setpDerivatives(i, *f2);
          res.setpAdjPtrs(i, *f1);
        }

        res.myValue = val;
        return res;
    }

    /// computes the dot product with T iterators
    template<class I1, class I2>
    static T dot_product_identical(I1 f1, I1 l1, I2 f2){
        static_assert(is_it_value_type<I1, T>::value,
                      "First iterator not to T");
        static_assert(is_it_value_type<I2, T>::value,
                      "Second iterator not to T");

        T res;
        const size_t n{static_cast<size_t>(std::distance(f1, l1))};
        res.createNode(2 * n);

        double val{};
        size_t i{};
        for(; f1 != l1; ++f1, ++f2, ++i){
            val += f1->value() * f2->value();
            res.setpDerivatives(i, f2->value());
            res.setpAdjPtrs(i, *f1);
            res.setpDerivatives(i + n, f1->value());
            res.setpAdjPtrs(i + n, *f2);
        }

        res.myValue = val;
        return res;
    }
};

} // namespace cfaad
