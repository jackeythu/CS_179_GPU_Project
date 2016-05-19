#ifndef KERNEL_DENSITY1D_ESTIMATOR_H__
#define KERNEL_DENSITY1D_ESTIMATOR_H__

#include <vector>
#include <string>

class kde1d{

public:
	kde1d(){};
	kde1d(double bandwidth, std::vector<double> data, std::string kernel_type_);
	~kde1d() = default;

	void add_data(std::vector<double> data);
	void add_data(double data);

	double pdf(double x);

	double get_bandwidth();
	void set_bandwidth(double bandwidth_);

private:
	double bandwidth;
	std::vector<double> data;
	double gauss_pdf(double x);
	std::string kernel_type;

};


#endif