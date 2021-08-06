
/*
Written by Antoine Savine in 2018

This code is the strict IP of Antoine Savine

License to use and alter this code for personal and commercial applications
is freely granted to any person or company who purchased a copy of the book

Modern Computational Finance: AAD and Parallel Simulations
Antoine Savine
Wiley, 2018

As long as this comment is preserved at the top of the file
*/

#pragma once

//  So we can instrument Gaussians like standard math functions
#include "gaussians.h"
#include <memory>
#include "AADConfig.h"
#include "AADReduction.h"
#include "AADNumWrapper.h"
#include "AADVectorFuncs.h"

namespace cfaad {

//  Routines for multi-dimensional AAD (chapter 14)
//  Set static context for multi-dimensional AAD

//	RAII: reset dimension 1 on destruction
struct numResultsResetterForAAD
{
	~numResultsResetterForAAD()
	{
#if AADMULTIOUT
		Tape::multi = false;
		Node::numAdj = 1;
#endif
	}
};

//  Routine: set dimension and get RAII resetter
inline auto setNumResultsForAAD(const bool multi = false, const size_t numResults = 1)
{
#if AADMULTIOUT
	Tape::multi = multi;
	Node::numAdj = numResults;
	return std::make_unique<numResultsResetterForAAD>();
#endif
}

//  Other utilities

//	Put collection on tape
template <class IT>
inline void putOnTape(IT begin, IT end)
{
    for_each(begin, end, [](Number& n) { n.putOnTape(); });
}

//	Convert collection between double and Number or reverse
template<class It1, class It2>
inline void convertCollection(It1 srcBegin, It1 srcEnd, It2 destBegin)
{
    using destType = std::remove_reference_t<decltype(*destBegin)>;
    transform(srcBegin, srcEnd, destBegin,
        [](const auto& source) { return destType(source); });
}

} // namespace cfaad