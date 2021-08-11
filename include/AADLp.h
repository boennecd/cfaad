#pragma once

#include "AADTape.h"
#include <memory>
#include <string>
#include <stdexcept>

#if AADLAPACK

namespace cfaad {
extern "C" {
    /// computes the Choleksy factorization
    void dpptrf_
    (const char * /* uplo */, const int * /* n */, double * /* ap */, 
     int * /* info */ /* TODO: handle the length of the char argument */);
     
    /// Either performs a forward or backward solve
    void dtpsv_
    (const char * /* uplo */, const char * /* trans */, const char * /* diag */,
     const int * /* n */, const double * /* ap */, double * /* x */, 
     const int * /* incx */
     /* TODO: handle the length of the char arguments */);
}

/// returns working memory of a given size
double *getLPWKMem(const size_t);
    
/// holds the Choleksy factorization of U^TU = X for a postive definite matrix
struct CholFactorization {
    /// the dimension 
    int n;
    /// the factorization
    std::unique_ptr<double[]> factorization{new double[(n * (n + 1)) / 2]};
    
    template<class I>
    CholFactorization(I begin, const int n): n(n)
    {
        {
            double * f = factorization.get();
            for(int j = 0; j < n; ++j)
                for(int i = 0; i <= j; ++i)
                    *f++ = begin[i + j * n];
        }
        
        int info{};
        char uplo{'U'};
        dpptrf_(&uplo, &n, factorization.get(), &info);
        
        if(info != 0)
            throw std::runtime_error
                ("dpptrf failed with code " + std::to_string(info));
    }
    
    /// computes either Ux = y or U^Tx = y
    void solveU(double *x, const bool trans) const {
        char uplo{'U'}, 
          c_trans = trans ? 'T' : 'N',
             diag{'N'};
        int incx{1};
        
        dtpsv_(&uplo, &c_trans, &diag, &n, factorization.get(), x, &incx);
    }
    
    /// computes U^TUx = y
    void solve(double *x) const {
        solveU(x, true);
        solveU(x, false);
    }
    
    /// helper class to create an object
    template<class I, class V>
    struct get_chol_factorization {
        /// the general case
        static CholFactorization get(I begin, const int n){
            double * wk_mem{getLPWKMem(static_cast<size_t>(n * n))};
            for(int j = 0; j < n; ++j)
                for(int i = 0; i <= j; ++i)
                    wk_mem[i + j * n] = begin[i + j * n].value();
                    
            return CholFactorization(wk_mem, n);
        }
    };
    
    template<class I>
    struct get_chol_factorization<I, double> {
        /// the special case with an iterator to doubles
        static CholFactorization get(I begin, const int n){
            return CholFactorization(begin, n);
        }
    };
    
    /// returns the Choleksy factorization
    template<class I>
    static CholFactorization getFactorization(I begin, const int n){
        return get_chol_factorization<I, it_value_type<I> >::get(begin, n);
    }
};

} // namespace cfaad

#endif // if AADLAPACK