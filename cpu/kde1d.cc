#include <vector>
#include <math.h>
#include <exception>
#include "kde1d.h"

#define PI 3.141592653

kde1d::kde1d(double bandwidth_, std::vector<double> data_){
	
	bandwidth = bandwidth_;
	if(data_.size() == 0){
		throw std::domain_error("Sample data error: no sample data. ");
	}

	for(int i = 0; i < data_.size(); ++i) data.push_back(data_[i]);
}

void kde1d::add_data(std::vector<double> data_){
	if(data_.size() == 0) return;
	for(int i = 0; i < data_.size(); ++i){
		data.push_back(data_[i]);
	}
}

void kde1d::add_data(double data_){
	data.push_back(data_);
}

double kde1d::pdf(double x){
	double result = 0;
	if(data.size() == 0){
		throw std::domain_error("Sample data error: no sample data for KDE. ");
	}

	for(int i = 0; i < data.size(); ++i){
		result += gauss_pdf((x - data[i])/bandwidth);
	}
	result = result / (bandwidth * data.size());
	return result;
}

double kde1d::get_bandwidth(){
	return bandwidth;
}

void kde1d::set_bandwidth(double bandwidth_){
	bandwidth = bandwidth_;
}

double kde1d::gauss_pdf(double x){
	double mean = 0.0;
	double std = 1.0;
	return 1.0/(std * sqrt(2*PI)) * exp(-(x-mean)*(x-mean) / (2*std*std));
}

