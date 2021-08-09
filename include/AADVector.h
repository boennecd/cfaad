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
        static_assert(is_it_value_type<I, T>::value, "Iterator is not to Ts");

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
                      "First iterator is not to Ts");
        static_assert(!is_it_value_type<I2, T>::value,
                      "Second iterator is to Ts");

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
                      "First iterator is not to Ts");
        static_assert(is_it_value_type<I2, T>::value,
                      "Second iterator is not to Ts");

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
    
    /*
     * computes the matrix vector product X.a where X is a m x n matrix and 
     * a is a n vector. The result is stored in the last argument which needs
     * to be iterator with space for at least m elements. The matrix is in 
     * column-major order.
     * 
     * This is the version where the matrix is a T type whereas the vector is 
     * for non-Ts. The last argument sets whether it is the X^T rather than X.
     */
    template<class I1, class I2, class I3>
    static void mat_vec_prod_TMat
    (I1 xf, I1 xl, I2 af, I2 al, I3 of, bool trans){
        static_assert(is_it_value_type<I1, T>::value,
                      "First iterator is not to Ts");
        static_assert(!is_it_value_type<I2, T>::value,
                      "Second iterator is to Ts");
        static_assert(is_it_value_type<I3, T>::value,
                      "Third iterators is not to Ts");
                      
        const size_t n = static_cast<size_t>(std::distance(af, al)), 
                     m = static_cast<size_t>(std::distance(xf, xl)) / n;
                     
        if(trans){
            for(size_t i = 0; i < m; ++i, ++of){
                of->createNode(n);
                of->myValue = 0;
                for(size_t j = 0; j < n; ++j, ++xf){
                    of->myValue += xf->value() * af[j];
                    of->setpDerivatives(j, af[j]);
                    of->setpAdjPtrs(j, *xf);
                }
            }
            return;
        }
                     
        for(auto v = of; v != of + m; ++v){
            v->createNode(n);
            v->myValue = 0;
        }
        
        for(size_t j = 0; j < n; ++j, ++af)
            for(size_t i = 0; i < m; ++i, ++xf){
                of[i].myValue += xf->value() * *af;
                of[i].setpDerivatives(j, *af);
                of[i].setpAdjPtrs(j, *xf);
            }
    }
    
    /**
     * the same as mat_vec_prod_TMat but where the first argument is for 
     * non-Ts and the second argument is for Ts.
     */
    template<class I1, class I2, class I3>
    static void mat_vec_prod_TVec
    (I1 xf, I1 xl, I2 af, I2 al, I3 of, bool trans){
        static_assert(!is_it_value_type<I1, T>::value,
                      "First iterator is to Ts");
        static_assert(is_it_value_type<I2, T>::value,
                      "Second iterator is not to Ts");
        static_assert(is_it_value_type<I3, T>::value,
                      "Third iterators is not to Ts");
                      
        const size_t n = static_cast<size_t>(std::distance(af, al)), 
                     m = static_cast<size_t>(std::distance(xf, xl)) / n;
                     
                     
        if(trans){
            for(size_t i = 0; i < m; ++i, ++of){
                of->createNode(n);
                of->myValue = 0;
                for(size_t j = 0; j < n; ++j, ++xf){
                    of->myValue += *xf  * af[j].value();
                    of->setpDerivatives(j, *xf);
                    of->setpAdjPtrs(j, af[j]);
                }
            }
            
            return;
        }
                     
        for(auto v = of; v != of + m; ++v){
            v->createNode(n);
            v->myValue = 0;
        }
        
        for(size_t j = 0; j < n; ++j, ++af)
            for(size_t i = 0; i < m; ++i, ++xf){
                of[i].myValue += *xf  * af->value();
                of[i].setpDerivatives(j, *xf);
                of[i].setpAdjPtrs(j, *af);
            }
    }
    
    /// the same as mat_vec_prod_TMat but where both iterators are for Ts.
    template<class I1, class I2, class I3>
    static void mat_vec_prod_identical
    (I1 xf, I1 xl, I2 af, I2 al, I3 of, bool trans){
        static_assert(is_it_value_type<I1, T>::value,
                      "First iterator is not to Ts");
        static_assert(is_it_value_type<I2, T>::value,
                      "Second iterator is not to Ts");
        static_assert(is_it_value_type<I3, T>::value,
                      "Third iterators is not to Ts");
                      
        const size_t n = static_cast<size_t>(std::distance(af, al)), 
                     m = static_cast<size_t>(std::distance(xf, xl)) / n;
                     
        if(trans){
            for(size_t i = 0; i < m; ++i, ++of){
                of->createNode(2 * n);
                of->myValue = 0;
                for(size_t j = 0; j < n; ++j, ++xf){
                    of->myValue += xf->value() * af[j].value();
                    of->setpDerivatives(j, xf->value());
                    of->setpAdjPtrs(j, af[j]);
                    of->setpDerivatives(j + n, af[j].value());
                    of->setpAdjPtrs(j + n, *xf);
                }
            }
            
            return;
        }
                     
        for(auto v = of; v != of + m; ++v){
            v->createNode(2 * n);
            v->myValue = 0;
        }

        for(size_t j = 0; j < n; ++j, ++af)
            for(size_t i = 0; i < m; ++i, ++xf){
                of[i].myValue += xf->value() * af->value();
                of[i].setpDerivatives(j, xf->value());
                of[i].setpAdjPtrs(j, *af);
                of[i].setpDerivatives(j + n, af->value());
                of[i].setpAdjPtrs(j + n, *xf);
            }
    }
};

} // namespace cfaad
