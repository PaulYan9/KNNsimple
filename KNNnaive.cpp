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

distance knn::KNNnaive::get_dist(int k)
{
	if (k > this->__index.size())
		k = this->__index.size();

	distance dist(k);
	std::transform(this->__index.begin(), this->__index.begin() + k, dist.begin(),
		[&](int i) {return this->__data->dis[i]; });

	return dist;
}

vecop::class_label knn::KNNnaive::get_labels(int k)
{
	if (k > this->__index.size())
		k = this->__index.size();

	vecop::class_label labels(k);
	std::transform(this->__index.begin(), this->__index.begin() + k, labels.begin(),
		[&](int i) {return this->__data->__Y[i]; });

	return dist;
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
	std::map<int, int> uniq_vals;

	for (auto y_pred : y)
	{
		if (uniq_vals.find(y_pred) == uniq_vals.end())
			uniq_vals[y_pred] = 1;
		else
			uniq_vals[y_pred] += 1;
	}

	auto max_el = std::max_element(uniq_vals.begin(), uniq_vals.end());

	return max_el->first;
}
