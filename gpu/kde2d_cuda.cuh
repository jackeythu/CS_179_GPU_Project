#ifndef __KDE2D_CUDA_H_
#define __KDE2D_CUDA_H_

void CallKDE2dKernel(unsigned int threadsPerBlock, unsigned int max_Number_of_block, float* dev_sample_x, float* dev_sample_y, int len_sample, float* dev_input_x, float* dev_input_y, float* dev_output, int len_input, float bandwidth);

#endif