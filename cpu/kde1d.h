// This class deals with one dimensional kernel density estimation.
// It supports both naive and adaptive version, and provides three kernel types.

#ifndef KERNEL_DENSITY1D_ESTIMATOR_H__
#define KERNEL_DENSITY1D_ESTIMATOR_H__

#include <vector>
#include <string>

class kde1d{

public:

	// Constructor:
	kde1d(){};

	// The initialization should provides default bandwidth, 
	// sample data, kernel type and which version(naive/adaptive). 
	kde1d(double bandwidth, std::vector<double> data, 
		std::string kernel_type_, bool adaptive_);
	~kde1d() = default;

	// These two functions are used to add more sample data points 
	// when using this class.
	void add_data(std::vector<double> data);
	void add_data(double data);

	// This function returns a naive version KDE
	double pdf_naive(double x);

	// This function returns adaptive KDE, in which the adaptive bandwidth
	// is propotional to naive estimate in sample points.
	double pdf_adaptive(double x);

	// This function returns the estimated pdf of a given point. It will automatically
	// calll pdf_naive or pdf_adaptive depending on user's need.
	double pdf(double x);

	// As an interface to bandwidth, these two functions can be used for cross validation
	// to obtain optimal results.
	double get_bandwidth();
	void set_bandwidth(double bandwidth_);

private:
	double bandwidth;
	std::vector<double> data;
	double gauss_pdf(double x);
	std::string kernel_type;
	
	// related to adaptive
	bool adaptive = false;

	double geometry_mean = 1.0;
	std::vector<double> pilot_density;
	void pilot_init();



};


#endif