/**
 * adds functions to perform a reduction by applying a function on one or two
 * iterators. This may be useful with expression template. There is some
 * overhead in some cases (the run time versions) because of a switch statement.
 *
 * TODO: using loops seems to be faster despite the larger tapes... See 
 *       ADDVector.h and AADVectorFuncs.h for better alternatives.
 */

#pragma once

#include <type_traits>

namespace cfaad {
template<class T>
using iterator_value_type_ref =
    typename std::add_lvalue_reference<
    typename std::iterator_traits<T>::value_type>::type;

using ReductionInitDef = double;

template <size_t N, size_t n, class INIT, class LHS, class OP>
struct UnaryReductionExpression {
   using NextTermT = UnaryReductionExpression<N, n - 1, INIT, LHS, OP>;
   using OPReturnT = typename std::result_of
       <OP(iterator_value_type_ref<LHS>)>::type;
   using ReturnT = decltype(
       std::declval<OPReturnT>() + std::declval<typename NextTermT::ReturnT>());

   static ReturnT eval(const LHS lhs, const OP &op, const INIT &init){
       static_assert(N >= n, "n > N");
       return op(*lhs) + NextTermT::eval(lhs + 1, op, init);
   }
};

template <size_t N, class INIT, class LHS, class OP>
struct UnaryReductionExpression<N, 0, INIT, LHS, OP>{
   using ReturnT = INIT;

   static INIT eval(const LHS &lhs, const OP &op, const INIT &init){
       return init;
   }
};

template<size_t N, class INIT = ReductionInitDef, class LHS, class OP>
typename UnaryReductionExpression
<N, N, INIT, LHS, OP>::ReturnT
UnaryReduction(const LHS lhs, const OP &op, const INIT &init = INIT{}){
   return UnaryReductionExpression
       <N, N, INIT, LHS, OP>::eval(lhs, op, init);
}

template<class BASE, class LHS, class OP>
class UnaryReductionRuntimeExpression {
    template<size_t N>
    using ReturnTN =
        typename UnaryReductionExpression
        <N, N, BASE, LHS, OP>::ReturnT;

    template<size_t N>
    static ReturnTN<N> loop_body
    (const LHS lhs, const OP &op, const BASE &init){
        return UnaryReductionExpression
            <N, N, BASE, LHS, OP>::eval(lhs, op, init);
    }

public:
    static BASE eval(LHS lhs, const size_t N,
                     const OP &op, BASE out = BASE{}){
        if(N == 0)
            // may be started at N == 0
            return loop_body<0>(lhs, op, out);

        for(size_t i = 0; i < N;){
            switch(N - i){
                case 1:
                out = loop_body<1>(lhs, op, out);
                ++i;
                ++lhs;
                break;

                case 2:
                out = loop_body<2>(lhs, op, out);
                i += 2;
                lhs += 2;
                break;

                case 3:
                out = loop_body<3>(lhs, op, out);
                i += 3;
                lhs += 3;
                break;

                case 4:
                case 5:
                case 6:
                case 7:
                out = loop_body<4>(lhs, op, out);
                i += 4;
                lhs += 4;
                break;

                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                out = loop_body<8>(lhs, op, out);
                i += 8;
                lhs += 8;
                break;

                default:
                out = loop_body<16>(lhs, op, out);
                i += 16;
                lhs += 16;
            }
        }

        return out;
    }
};

template<class BASE = ReductionInitDef, class LHS, class OP>
BASE UnaryReduction
(const LHS lhs, const size_t N,
 const OP &op, const BASE &init = BASE{}){
    return UnaryReductionRuntimeExpression<BASE, LHS, OP>::eval(
        lhs, N, op, init);
}

/// Binary reduction

template <size_t N, size_t n, class INIT, class LHS,
          class RHS, class OP>
struct BinaryReductionExpression {
   using NextTermT = BinaryReductionExpression<N, n - 1, INIT, LHS, RHS, OP>;
   using OPReturnT = typename std::result_of
       <OP(iterator_value_type_ref<LHS>, iterator_value_type_ref<RHS>)>::type;
   using ReturnT = decltype(
       std::declval<OPReturnT>() + std::declval<typename NextTermT::ReturnT>());

   static ReturnT eval(LHS &lhs, RHS &rhs,
                       const OP &op, const INIT &init){
       static_assert(N >= n, "n > N");
       auto term = op(*lhs++, *rhs++);
       return term + NextTermT::eval(lhs, rhs, op, init);
   }
};

template <size_t N, class INIT, class LHS, class RHS,
          class OP>
struct BinaryReductionExpression<N, 0, INIT, LHS, RHS, OP>{
   using ReturnT = INIT;

   static INIT eval(const LHS &lhs, const RHS &rhs,
                    const OP &op, const INIT &init){
       return init;
   }
};

template<size_t N, class INIT = ReductionInitDef, class LHS,
         class RHS, class OP>
typename BinaryReductionExpression
<N, N, INIT, LHS, RHS, OP>::ReturnT
BinaryReduction(LHS lhs, RHS rhs,
                const OP &op, const INIT &init = INIT{}){
   return BinaryReductionExpression
       <N, N, INIT, LHS, RHS, OP>::eval(lhs, rhs, op, init);
}

template<class BASE, class LHS, class RHS, class OP>
class BinaryReductionRuntimeExpression {
    template<size_t N>
    using ReturnTN =
        typename BinaryReductionExpression
        <N, N, BASE, LHS, RHS, OP>::ReturnT;

    template<size_t N>
    static ReturnTN<N> loop_body
    (const LHS lhs, const RHS rhs,
     const OP &op, const BASE &init){
        LHS l = lhs;
        RHS r = rhs;
        return BinaryReductionExpression
            <N, N, BASE, LHS, RHS, OP>::eval(l, r, op, init);
    }

public:
    static BASE eval(LHS lhs, RHS rhs, const size_t N,
                     const OP &op, BASE out = BASE{}){
        if(N == 0)
            // may be started at N == 0
            return loop_body<0>(lhs, rhs, op, out);

        for(size_t i = 0; i < N;){
            switch(N - i){
                case 1:
                out = loop_body<1>(lhs, rhs, op, out);
                ++i;
                ++lhs;
                ++rhs;
                break;

                case 2:
                out = loop_body<2>(lhs, rhs, op, out);
                i += 2;
                lhs += 2;
                rhs += 2;
                break;

                case 3:
                out = loop_body<3>(lhs, rhs, op, out);
                i += 3;
                lhs += 3;
                rhs += 3;
                break;

                case 4:
                case 5:
                case 6:
                case 7:
                out = loop_body<4>(lhs, rhs, op, out);
                i += 4;
                lhs += 4;
                rhs += 4;
                break;

                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                case 15:
                out = loop_body<8>(lhs, rhs, op, out);
                i += 8;
                lhs += 8;
                rhs += 8;
                break;

                default:
                out = loop_body<16>(lhs, rhs, op, out);
                i += 16;
                lhs += 16;
                rhs += 16;
            }
        }

        return out;
    }
};

template<class BASE = ReductionInitDef, class LHS, class RHS, class OP>
BASE BinaryReduction
(const LHS lhs, const RHS rhs, const size_t N,
 const OP &op, const BASE &init = BASE{}){
    return BinaryReductionRuntimeExpression<BASE, LHS, RHS, OP>::eval(
        lhs, rhs, N, op, init);
}

} // namespace cfaad
