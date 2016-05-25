#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

#include <cuda_runtime.h>
#include <algorithm>
#include "kde_cuda.cuh"

using namespace std;

#define uniform 0
#define gaussian 1
#define quartic 2

int main(int argc, char* argv[]){

    if(argc != 3){
        printf("Usage: (threads per block) (max number of blocks)\n");
        exit(-1);
    }
    int threadsPerBlock = atoi(argv[1]);
    int max_Number_of_block = atoi(argv[2]);


    //input data
    //observation data is stored in data, and then copied to data_host
    //predicted data is created in input, and then copied to output_host. output_host is to store Prob(i) further.
    //which means, x-axis is input, and y-axis is output_host
	ifstream myfile;
	myfile.open("../data/gauss_single.csv");
	myfile.is_open() ? cout << "open data file successfully" : cout << "Error: Cannot open data file!";
	cout << endl;

	vector<double> data;
    float x = 0;
    while(myfile >> x){
    	data.push_back(x);
    }
    int len_data = data.size();
    float* data_host = new float[len_data];
    for(int i = 0; i < len_data; ++i){
    	data_host[i] = data[i];
    }

    vector<double> input;
    for(int i = 0; i < 10000; ++i){
        input.push_back((i - 5000) * 1.0 / 500 + 5);
    }
    int len_input = input.size();
    float* output_host = new float[len_input];
    for(int i = 0; i < len_input; ++i){
        output_host[i] = input[i];
    }
    cout << "finish inputting data" << endl;

    //allocate memory for gpu and cpu
    //output_host is to store probability result
    float* dev_data;
    float* dev_output;

    float time_initial, time_final;
    time_initial = clock();
    cudaMalloc((void**) &dev_data, len_data * sizeof(float));
    cudaMalloc((void**) &dev_output, len_input * sizeof(float));
    cudaMemcpy(dev_data, data_host, len_data * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_output, output_host, len_input * sizeof(float), cudaMemcpyHostToDevice);
    CallKDEKernel(threadsPerBlock, max_Number_of_block, dev_data, dev_output, len_data, len_input, 1.0, uniform);
    cudaMemcpy(output_host, dev_output, len_input * sizeof(float), cudaMemcpyDeviceToHost);
    time_final = clock();

    cout << "cost time is " << (time_final - time_initial)/CLOCKS_PER_SEC << endl;
    //output data to file
    ofstream outfile2;
    outfile2.open("../data/kde_gauss_single_output_gpu.csv");
    for(int i = 0; i < len_input; ++i){
        outfile2 << input[i] << "," << output_host[i];
        outfile2 << "\n";
    }
    return 0;


}
