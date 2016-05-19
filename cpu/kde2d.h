#ifndef KERNEL2d_DENSITY_ESTIMATOR_H__
#define KERNEL2d_DENSITY_ESTIMATOR_H__

#include <vector>
#include <string>
#include "Point2d.h"

class kde2d{

public:
	kde2d(){};
	kde2d(double bandwidth, std::vector<Point2d> data, std::string kernel_type_);
	~kde2d() = default;

	void add_data(std::vector<Point2d> data);
	void add_data(Point2d data);

	double pdf(Point2d x);

	double get_bandwidth();
	
	void set_bandwidth(double bandwidth_);

private:
	double bandwidth;
	std::vector<Point2d> data;
	std::string kernel_type;
	double gauss_pdf(Point2d x);

};


#endif