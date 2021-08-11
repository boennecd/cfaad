#pragma once

#include "AADTape.h"
#include <memory>
#include <string>
#include <stdexcept>
#include <algorithm>

#if AADLAPACK

namespace cfaad {
extern "C" {
    /// computes the Choleksy factorization
    void dpptrf_
    (const char * /* uplo */, const int * /* n */, double * /* ap */, 
     int * /* info */ /* TODO: handle the length of the char argument */);
     
    /// either performs a forward or backward solve
    void dtpsv_
    (const char * /* uplo */, const char * /* trans */, const char * /* diag */,
     const int * /* n */, const double * /* ap */, double * /* x */, 
     const int * /* incx */ 
     /* TODO: handle the length of the char arguments */);
     
    /// computes the inverse from the Choleksy factorization
    void dpptri_
    (const char * /* uplo */, const int * /* n */, double *ap, 
     int * /* info */ /* TODO: handle the length of the char arguments */);
}

/// returns working memory of a given size
double *getLPWKMem(const size_t);
    
/// holds the Choleksy factorization of U^TU = X for a postive definite matrix
class CholFactorization {
public:
    /// the dimension 
    const int n;
    /// default value for comp inv
    static constexpr bool def_comp_inv{true};
    
    template<class I>
    CholFactorization(I begin, const int n, const bool comp_inv = def_comp_inv): 
    n{n}, 
    mem{new double[comp_inv ? n * (n + 1) : (n * (n + 1)) / 2]},
    factorization{mem.get()},
    inverse{comp_inv ? factorization + (n * (n + 1))/2 : nullptr}
    {
        {
            double * f{factorization};
            for(int j = 0; j < n; ++j)
                for(int i = 0; i <= j; ++i)
                    *f++ = begin[i + j * n];
        }
        
        // compute the factorization
        int info{};
        char uplo{'U'};
        dpptrf_(&uplo, &n, factorization, &info);
        
        if(info != 0)
            throw std::runtime_error
                ("dpptrf failed with code " + std::to_string(info));
                
        if(!comp_inv)
            return;
                
        // compute the inverse
        std::copy(factorization, inverse, inverse);
        dpptri_(&uplo, &n, inverse, &info);
        if(info != 0)
            throw std::runtime_error
                ("dpptri failed with code " + std::to_string(info));
    }
    
    /// returns a pointer to the inverse (upper triangle)
    double const * get_inv() const {
        return inverse;
    }
    
    /// computes the determinant of the original matrix
    double determinant() const {
        double out{1};
        const double *v{factorization};
        for(int i = 0; i < n; ++i, v += i + 1)
            out *= *v;
        return out * out;
    }
    
    /// computes either Ux = y or U^Tx = y
    void solveU(double *x, const bool trans) const {
        char uplo{'U'}, 
          c_trans = trans ? 'T' : 'N',
             diag{'N'};
        int incx{1};
        
        dtpsv_(&uplo, &c_trans, &diag, &n, factorization, x, &incx);
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
        static CholFactorization get(I begin, const int n, bool const comp_inv){
            double * wk_mem{getLPWKMem(static_cast<size_t>(n * n))};
            for(int j = 0; j < n; ++j)
                for(int i = 0; i <= j; ++i)
                    wk_mem[i + j * n] = begin[i + j * n].value();
                    
            return CholFactorization(wk_mem, n, comp_inv);
        }
    };
    
    template<class I>
    struct get_chol_factorization<I, double> {
        /// the special case with an iterator to doubles
        static CholFactorization get(I begin, const int n, const bool comp_inv){
            return CholFactorization(begin, n, comp_inv);
        }
    };
    
    /// returns the Choleksy factorization
    template<class I>
    static CholFactorization getFactorization
    (I begin, const int n, const bool comp_inv = def_comp_inv){
        return get_chol_factorization<I, it_value_type<I> >
            ::get(begin, n, comp_inv);
    }

private:
    /// memory
    std::unique_ptr<double[]> mem;
    /// the factorization
    double * const factorization;
    /// the inverse (in upper triangle)
    double * const inverse;
};

} // namespace cfaad

#endif // if AADLAPACK