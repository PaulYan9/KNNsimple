#include "KNNnaive.h"

using namespace knn;

KNNnaive::KNNnaive(int k, DistMethod method) : __k(k), __method(method)
{
	this->__data = new __data_t();
}

KNNnaive::~KNNnaive()
{
	delete this->__data;
}

void KNNnaive::fit(const vecop::features& X, const vecop::class_label& Y)
{
	this->__data->__X = X;
	this->__data->__Y = Y;
	this->__data->dis.clear();
}

void KNNnaive::predict(vecop::feature X)
{
	this->__data->dis.clear();
	this->__index.clear();
	if (this->__method == DistMethod::EUCLIDEAN_DIST)
		this->__euclidean_dist(X);
	else if (this->__method == DistMethod::MANHATTAN_DIST)
		this->__manhattan_dist(X);

	for (int i = 0; i < this->__data->dis.size(); i++)
		this->__index.push_back(i);
	
	std::sort(this->__index.begin(), this->__index.end(), [](int a, int b)
		{return this->__data->dis[a] < this->__data->dis[b]; });

	vecop::class_label y_pred(this->__k);
	std::transform(this->__index.begin(), this->__index.begin() + this->__k, y-pred.begin(), 
		[&](int i) {return this->__data->__Y[i]; });

	return this->__max_repeat(y_pred);
}

inline void knn::KNNnaive::__euclidean_dist(const vecop::feature& obj)
{
	for (auto feat : this->__data->__X)
		this->__data->dis.push_back(dist::euclidean_dist(feat, obj));
}

inline void knn::KNNnaive::__manhattan_dist(const vecop::feature& obj)
{
	for (auto feat : this->__data->__X)
		this->__data->dis.push_back(dist::manhattan_dist(feat, obj));
}

inline int knn::KNNnaive::__max_repeat(const vecop::class_label& y)
{
	return 0;
}
