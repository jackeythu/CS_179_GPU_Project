#include <vector>
#include <math.h>
#include <string>
#include <exception>
#include "kde2d.h"
#include "Point2d.h"

#define PI 3.141592653

kde2d::kde2d(double bandwidth_, std::vector<Point2d> data_, std::string kernel_type_){

	bandwidth = bandwidth_;
	
	if(data_.size() == 0){
		throw std::domain_error("Sample data error: no sample data. ");
	}

	if(kernel_type_ != "gaussian" and kernel_type_ != "uniform" and kernel_type_ != "quartic"){
		throw std::domain_error("Kernel error: unsupported kernel. ");
	}	
	for(int i = 0; i < data_.size(); ++i) data.push_back(data_[i]);

	kernel_type = kernel_type_;
}


void kde2d::add_data(std::vector<Point2d> data_){
	if(data_.size() == 0) return;
	for(int i = 0; i < data_.size(); ++i){
		data.push_back(data_[i]);
	}
}

void kde2d::add_data(Point2d data_){
	data.push_back(data_);
}

double kde2d::pdf(Point2d x){
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

double kde2d::get_bandwidth(){
	return bandwidth;
}

void kde2d::set_bandwidth(double bandwidth_){
	bandwidth = bandwidth_;
}

double kde2d::gauss_pdf(Point2d x){
	if(kernel_type != "gaussian" and kernel_type != "uniform" and kernel_type != "quartic"){
		throw std::domain_error("Kernel error: unsupported kernel. ");
	}
	else if(kernel_type == "gaussian"){

		return 1.0/(2*PI) * exp(-x.norm2() / 2.0);
	}
	else if(kernel_type == "uniform"){
		if(x.norm() > 1) return 0;
		else return 1.0/2;
	}
	else{
		if(x.norm() > 1) return 0;
		else return 15.0/16 * (1-x.norm2())*(1-x.norm2());
	}
	
}


