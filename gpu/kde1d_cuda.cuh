#ifndef __KDE_CUDA_H_
#define __KDE_CUDA_H_


void CallKDEKernel(unsigned int threadsPerBlock, unsigned int max_Number_of_block, float* dev_data, float* dev_output, int len_data, int len_input, float bandwidth, int kernel_type);

#endif
