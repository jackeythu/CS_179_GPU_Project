#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "kde1d_cuda.cuh"

#define PI 3.141592653


// Three kernel types supported 
__device__ float gausskernel(float data){
	return 1.0/(1.0 * sqrt(2*PI)) * expf(-data*data / 2.0);
}

__device__ float uniformkernel(float data){
	if(fabs(data) > 1) return 0;
	else return 1.0/2;
}

__device__ float quartickernel(float data){
	if(fabs(data) > 1) return 0; 
	else return 15.0/16 * (1-data*data)*(1-data*data);
}

__device__ float kernel(float data, int kernel_type){
	if(kernel_type == 0){
		return uniformkernel(data);
	}
	else if(kernel_type == 1){
		return gausskernel(data);
	}
	else return quartickernel(data);
}


// THis kernel deals with naive 1D kde.
__global__
void KDEkernel(float* dev_sample, float* dev_input, float* dev_output, int len_sample, int len_data, float bandwidth, int kernel_type){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < len_data){
		float sum_ker = 0;
		register float dev_input_idx = dev_input[idx];
		for(int i = 0; i < len_sample; ++i){
			sum_ker += kernel((dev_input_idx-dev_sample[i])/bandwidth, kernel_type) / bandwidth;
		}
		dev_output[idx] = sum_ker/len_sample;
		
		idx += blockDim.x * gridDim.x;
	}
}



// This kernel deals with adaptive 1D kde.
// default dev_estimate_sample is the copy of dev_sample
__global__
void KDEadaptiveKernel(float* dev_sample, float* dev_estimate_sample, float* dev_input, float* dev_output, int len_sample, int len_data, float bandwidth, int kernel_type){
	
	__shared__ float shr_sample[1024];
	__shared__ float shr_estimate_sample[1024];
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// calculate geometry mean
	float G = 1.0;
	for(int i = 0; i < len_sample; ++i){
		G = powf(G, i*1.0/(i+1)) * powf(dev_estimate_sample[i], 1.0/(i+1));
	}
	
	// estimate inputs using adaptive version	
	while(idx < len_data){
		float sum_ker = 0;
		register float dev_input_idx = dev_input[idx];
		float lambda;
		for(int i = 0; i < len_sample-1024; i = i + 1024){

			for(int j = 0; j < 1024; ++j){
				shr_sample[threadIdx.x + j] = dev_sample[j + (threadIdx.x+i)];
				shr_estimate_sample[threadIdx.x + j] = dev_estimate_sample[j + (threadIdx.x+i)];
			}

			for(int j = 0; j < 1024; ++j){
			lambda = powf(G/shr_estimate_sample[j + (threadIdx.x+i)], 0.5);
			sum_ker += kernel((dev_input_idx - shr_sample[j])/(bandwidth*lambda), kernel_type) /(bandwidth*lambda);
			}
		}
		dev_output[idx] = sum_ker/len_sample;

		idx += blockDim.x * gridDim.x;
	}
}


void CallKDEKernel(unsigned int threadsPerBlock, unsigned int max_Number_of_block, float* dev_sample, float* dev_input, float* dev_output, int len_sample, int len_data, float bandwidth, int kernel_type){
	KDEkernel<<<max_Number_of_block, threadsPerBlock>>>(dev_sample, dev_input, dev_output, len_sample, len_data, bandwidth, kernel_type);
}


void CallKDEadaptiveKernel(unsigned int threadsPerBlock, unsigned int max_Number_of_block, float* dev_sample, float* dev_estimate_sample, float* dev_input, float* dev_output, int len_sample, int len_data, float bandwidth, int kernel_type){
printf("in call function before kernel\n");
	KDEadaptiveKernel<<<max_Number_of_block, threadsPerBlock>>>(dev_sample, dev_estimate_sample, dev_input, dev_output, len_sample, len_data, bandwidth, kernel_type);
printf("in call function after kernel\n");
}


