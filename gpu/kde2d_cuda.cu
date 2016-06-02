#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "kde2d_cuda.cuh"

#define PI 3.141592653

__device__ float gausskernel(float distance2){
	return 1.0/(2*PI) * expf(-distance2 / 2.0);
}

__global__
void 
KDE2dKernel(float* dev_sample_x, float* dev_sample_y, int len_sample, float* dev_input_x, float* dev_input_y, float* dev_output, int len_input, float bandwidth){
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	while(idx < len_input){
		float sum_ker = 0;
		for(int i = 0; i < len_sample; ++i){
			float distance2 = powf(dev_input_x[idx] - dev_sample_x[i], 2) + powf(dev_input_y[idx] - dev_sample_y[i], 2);
			sum_ker += gausskernel(distance2/bandwidth);
		}
		dev_output[idx] = sum_ker / len_sample;

		idx += blockDim.x * gridDim.x;
	}
}



void CallKDE2dKernel(unsigned int threadsPerBlock, unsigned int max_Number_of_block, float* dev_sample_x, float* dev_sample_y, int len_sample, float* dev_input_x, float* dev_input_y, float* dev_output, int len_input, float bandwidth){
	KDE2dKernel<<<max_Number_of_block, threadsPerBlock>>>(dev_sample_x, dev_sample_y, len_sample, dev_input_x, dev_input_y, dev_output, len_input, bandwidth);
}
