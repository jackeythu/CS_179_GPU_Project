#include <vector>
#include <math.h>
#include <exception>
#include <string>
#include "kde1d.h"

#define PI 3.141592653

kde1d::kde1d(double bandwidth_, std::vector<double> data_, std::string kernel_type_){
	
	if(data_.size() == 0){
		throw std::domain_error("Sample data error: no sample data. ");
	}

	if(kernel_type_ != "gaussian" and kernel_type_ != "uniform" and kernel_type_ != "quartic"){
		throw std::domain_error("Kernel error: unsupported kernel. ");
	}	

	bandwidth = bandwidth_;

	for(int i = 0; i < data_.size(); ++i) data.push_back(data_[i]);

	kernel_type = kernel_type_;
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
	
	if(kernel_type != "gaussian" and kernel_type != "uniform" and kernel_type != "quartic"){
		throw std::domain_error("Kernel error: unsupported kernel. ");
	}
	else if(kernel_type == "gaussian"){
		double mean = 0.0;
		double std = 1.0;
		return 1.0/(std * sqrt(2*PI)) * exp(-(x-mean)*(x-mean) / (2*std*std));
	}
	else if(kernel_type == "uniform"){
		if(fabs(x) > 1) return 0;
		else return 1.0/2;
	}
	else{
		if(fabs(x) > 1) return 0;
		else return 15.0/16 * (1-x*x)*(1-x*x);
	}
}

