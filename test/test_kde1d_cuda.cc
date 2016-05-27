#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <exception>
#include <fstream>

#include <cuda_runtime.h>
#include <algorithm>
#include "kde1d_cuda.cuh"

using namespace std;

#define uniform 0
#define gaussian 1
#define quartic 2

void kde1d_cuda_naive(int threadsPerBlock, int max_Number_of_block, float* sample, int len_sample, float* input_host, float* output_host, int len_data, float bandwidth, int kernel_type){

    //input data
    //observation data is stored in data, and then copied to data_host
    //predicted data is created in input, and then copied to output_host. output_host is to store Prob(i) further.
    //which means, x-axis is input, and y-axis is output_host

    float* dev_sample;
    float* dev_input;
    float* dev_output;

    float time_initial, time_final;
    time_initial = clock();

    cudaMalloc((void**) &dev_sample, len_sample * sizeof(float));
    cudaMemcpy(dev_sample, sample, len_sample * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_input, len_data * sizeof(float));
    cudaMemcpy(dev_input, input_host, len_data * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &dev_output, len_data * sizeof(float));
    cudaMemcpy(dev_output, output_host, len_data * sizeof(float), cudaMemcpyHostToDevice);

    CallKDEKernel(threadsPerBlock, max_Number_of_block, dev_sample, dev_input, dev_output, len_sample, len_data, bandwidth, kernel_type);
    cudaMemcpy(output_host, dev_output, len_data * sizeof(float), cudaMemcpyDeviceToHost);
    time_final = clock();

    cout << "cost time is " << (time_final - time_initial)/CLOCKS_PER_SEC << endl;

}


int main(int argc, char* argv[]){

    if(argc != 8){
        printf("Usage: <threads per block> <max number of blocks> <sample file path> <kernel_type:uniform/gaussian/quartic> <range_min> <range_max> <output length>\n");
        exit(-1);
    }

    int threadsPerBlock = atoi(argv[1]);
    int max_Number_of_block = atoi(argv[2]);
    //std::string sample_path(argv[3]);
    std::string kernel_type_(argv[4]);
    int range_min = atoi(argv[5]);
    int range_max = atoi(argv[6]);
    int len_data = atoi(argv[7]);

    int kernel_type;
    if(kernel_type_ != "uniform" and kernel_type_ != "gaussian" and kernel_type_ != "quartic"){
        printf("Kernel error: unsupported kernel type.");
        exit(-1);
    }
    else if(kernel_type_ == "uniform"){
        kernel_type = uniform;
    }
    else if(kernel_type_ == "gaussian"){
        kernel_type = gaussian;
    }
    else{
        kernel_type = quartic;
    }



    // Input sample data from file and store in vector
	ifstream myfile;
	myfile.open(argv[3]);
	myfile.is_open() ? cout << "open data file successfully" : cout << "Error: Cannot open data file!";
	cout << endl;

    vector<float> sample_;
    float x = 0;
    while(myfile >> x){
        sample_.push_back(x);
    }

    int len_sample = sample_.size();
    float* sample = new float[len_sample];
    for(int i = 0; i < len_sample; ++i){
        sample[i] = sample_[i];
    }



    // Create and initialize input_host and output_host, the former is to give range of estimated points,
    // the latter is to store estimated probabilities from kde.
    // Both will be passed to GPU via dev_input and dev_output.
    float step = (range_max - range_min)*1.0 / len_data;
    float* input_host = new float[len_data];
    float* output_host = new float[len_data];
    for(int i = 0; i < len_data; ++i){
        input_host[i] = range_min + step * i;
        output_host[i] = 0;
    }


    // Call kde functions
    kde1d_cuda_naive(threadsPerBlock, max_Number_of_block, sample, len_sample, input_host, output_host, len_data, 1.0, kernel_type);

    //output data to file
    ofstream outfile2;
    outfile2.open("../data/kde1d_cuda_output.csv");
    for(int i = 0; i < len_data; ++i){
        outfile2 << input_host[i] << "," << output_host[i];
        outfile2 << "\n";
    }
    return 0;
}
