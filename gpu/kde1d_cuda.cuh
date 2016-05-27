#ifndef __KDE_CUDA_H_
#define __KDE_CUDA_H_


void CallKDEKernel(unsigned int threadsPerBlock, unsigned int max_Number_of_block, float* dev_sample, float* dev_input, float* dev_output, int len_sample, int len_data, float bandwidth, int kernel_type);

void CallKDEadaptiveKernel(unsigned int threadsPerBlock, unsigned int max_Number_of_block, float* dev_sample, float* dev_estimate_sample, float* dev_input, float* dev_output, int len_sample, int len_data, float bandwidth, int kernel_type);
#endif
