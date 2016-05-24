#include <vector>
#include <math.h>
#include <exception>
#include <string>
#include <iostream>
#include "kde1d.h"

#define PI 3.141592653

kde1d::kde1d(double bandwidth_, std::vector<double> data_, std::string kernel_type_, bool adaptive_){
	
	if(data_.size() == 0){
		throw std::domain_error("Sample data error: no sample data. ");
	}

	if(kernel_type_ != "gaussian" and kernel_type_ != "uniform" and kernel_type_ != "quartic"){
		throw std::domain_error("Kernel error: unsupported kernel. ");
	}	

	bandwidth = bandwidth_;

	for(int i = 0; i < data_.size(); ++i) data.push_back(data_[i]);

	// currently we support three kinds of kernel_types
	kernel_type = kernel_type_;

	// adaptive initialization
	adaptive = adaptive_;
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


double kde1d::pdf_naive(double x){
	double result = 0;
	if(data.size() == 0){
		throw std::domain_error("Sample data error: no sample data for naive KDE. ");
	}

	for(int i = 0; i < data.size(); ++i){
		result += gauss_pdf((x - data[i])/bandwidth);
	}
	result = result / (bandwidth * data.size());
	return result;
}


void kde1d::pilot_init(){
	if(data.size() == 0){
		throw std::domain_error("Sample data error: no sample data for adaptive KDE.");
	}

	// initialize pilot_density, which is to store pilot density estimate result.
	for(int i = 0; i < data.size(); ++i){
		pilot_density.push_back(pdf_naive(data[i]));
	}

	// initialize geometry_mean
	for(int i = 0; i < pilot_density.size(); ++i){
		geometry_mean *= pilot_density[i];
	}
	geometry_mean = pow(geometry_mean, 1.0/pilot_density.size());
}



double kde1d::pdf_adaptive(double x){
	double result = 0;
	if(data.size() == 0){
		throw std::domain_error("Sample data error: no sample data for adaptive KDE.");
	}

	if(pilot_density.size() == 0){
		pilot_init();
		std::cout << "geometry_mean is " << geometry_mean << std::endl;
		std::cout << "pilot_density0 is " << pilot_density[0] << std::endl;
	}

	double temp_bandwidth = 0;
	for(int i = 0; i < data.size(); ++i){
		temp_bandwidth = bandwidth * sqrt(geometry_mean/pilot_density[i]);
		result += gauss_pdf((x - data[i]) / temp_bandwidth) / temp_bandwidth;
	}
	result = result / data.size();
	return result;
}

double kde1d::pdf(double x){
	if(adaptive == false){
		return pdf_naive(x);
	}
	else{
		return pdf_adaptive(x);
	}
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

