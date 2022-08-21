#pragma once

#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

namespace vecop
{
	typedef std::vector<double> feature;
	typedef std::vector<feature> features;
	typedef std::vector<int> class_label;

	feature diff(const feature& X1, const feature& X2)
	{
		if (X1.size() != X2.size())
			throw("Different sizes");

		feature res(X1.size());
		for (int i = 0; i <= X1.size(); i++)
			res[i] = X1[i] - X2[i];

		return res;
	}

	double sum(const feature& X)
	{
		return std::accumulate(X.begin(), X.end(), 0);
	}

	feature abs(const feature& X)
	{
		feature res(X.size());
		std::transform(X.begin(), X.end(), res.begin(), [](double e) { return std::abs(e); });

	}

	feature power(const feature& X, int p)
	{
		feature res(X.size());
		std::transform(X.begin(), X.end(), res.begin(), [](double e) { return std::pow(e, 2); });
	}
}