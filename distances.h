#pragma once

#include "vecoperation.h"

namespace dist
{
	double euclidean_dist(vecop::feature X1, vecop::feature X2)
	{
		auto res = vecop::diff(X1, X2);
		res = vecop::power(res);
		auto distance = vecop::sum(res);

		return std::sqrt(distance);
	}

	double manhattan_dist(const vecop::feature& X1, const vecop::feature& X2)
	{
		auto res = vecop::diff(X1, X2);
		res = vecop::abs(res);

		return vecop::sum(res);
	}
}