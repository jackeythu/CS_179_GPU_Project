#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "kde_cuda.cuh"

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



__global__
void KDEkernel(float* dev_data, float* dev_output, int len_data, int len_input, int bandwidth, int kernel_type){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < len_input){
		float sum_ker = 0;
		for(int i = 0; i < len_data; ++i){
			sum_ker += kernel((dev_output[idx]-dev_data[i])/bandwidth, kernel_type) / bandwidth;
		}
		dev_output[idx] = sum_ker/len_data;
		
		idx += blockDim.x * gridDim.x;
	}
}




void CallKDEKernel(unsigned int threadsPerBlock, unsigned int max_Number_of_block, float* dev_data, float* dev_output, int len_data, int len_input, float bandwidth, int kernel_type){
	KDEkernel<<<max_Number_of_block, threadsPerBlock>>>(dev_data, dev_output, len_data, len_input, bandwidth, kernel_type);
}