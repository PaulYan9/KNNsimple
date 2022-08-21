#pragma once

#include <vector>
#include "vecoperation.h"

class KNNnaive
{
	private:
		vecop::features* __X;
		vecop::class_label* __Y;
		int __k;

	public:
		KNNnaive(int k = 3);

		void fit(const vecop::features& X, const vecop::class_label& Y);
		void predict(vecop::feature X);
};

