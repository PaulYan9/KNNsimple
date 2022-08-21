#pragma once

#include "vecoperation.h"
#include "distances.h"

namespace knn
{
	typedef std::vector<double> distance;

	enum class DistMethod
	{
		EUCLIDEAN_DIST,
		MANHATTAN_DIST
	};

	class KNNnaive
	{
		private:
			struct __data_t
			{
				vecop::features __X;
				vecop::class_label __Y;
				distance dis;
			};

			__data_t* __data;
			DistMethod __method;
			int __k;

		public:
			KNNnaive(int k = 3, DistMethod method = DistMethod::EUCLIDEAN_DIST);
			~KNNnaive();

			void fit(const vecop::features& X, const vecop::class_label& Y);
			void predict(vecop::feature X);
	};
}

