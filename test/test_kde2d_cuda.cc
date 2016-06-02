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
#include "../gpu/kde2d_cuda.cuh"
#include "Point2d.h"

using namespace std;


int main(){
    
    // sample data:
    vector<Point2d> sample;
    ifstream intputfile;
    intputfile.open("../data/gaussian2d.csv");
    double x;
    double y;
    while(intputfile>>x>>y){
        Point2d temp(x,y);
        sample.push_back(temp);
    }

    float* sample_x = new float[sample.size()];
    float* sample_y = new float[sample.size()];
    for(int i = 0; i < sample.size(); ++i){
        sample_x[i] = sample[i].x();
        sample_y[i] = sample[i].y();
    }
    cout << "input sample data successfully!" << endl;


    // input data
    int len_x = 100;
    int len_y = 100;
    int N = len_x * len_y;
    float min_x = -5;
    float max_x = 5;
    float min_y = -5;
    float max_y = 5;

    float* input_x = new float[N];
    float* input_y = new float[N];

    float step_x = (max_x - min_x) / len_x;
    float step_y = (max_y - min_y) / len_y;
    for(int i = 0; i < len_x; ++i){
        for(int j = 0; j < len_y; ++j){
            input_x[i*len_y + j] = min_x + step_x * i;
            input_y[i*len_y + j] = min_y + step_y * j;
        }
    }

    cout << "initialize input data successfully!" << endl;


    // malloc and memcpy to device memory

    // malloc sample data points
    float* dev_sample_x;
    float* dev_sample_y;
    cudaMalloc((void**)&dev_sample_x, sample.size() * sizeof(float));
    cudaMemcpy(dev_sample_x, sample_x, sample.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&dev_sample_y, sample.size() * sizeof(float));
    cudaMemcpy(dev_sample_y, sample_y, sample.size() * sizeof(float), cudaMemcpyHostToDevice);

    // malloc input data points
    float* dev_input_x;
    float* dev_input_y;
    cudaMalloc((void**)&dev_input_x, N * sizeof(float));
    cudaMemcpy(dev_input_x, input_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&dev_input_y, N * sizeof(float));
    cudaMemcpy(dev_input_y, input_y, N * sizeof(float), cudaMemcpyHostToDevice);

    // malloc output data points
    float* dev_output;
    cudaMalloc((void**)&dev_output, N * sizeof(float));
    cudaMemset(dev_output, 0, N * sizeof(float));




    // Call cuda kernel
    int threadPerblock = 300;
    int max_number_of_block = 300;
    
    CallKDE2dKernel(threadPerblock, max_number_of_block, dev_sample_x, dev_sample_y, sample.size(), dev_input_x, dev_input_y, dev_output, N, 10);
    
    float* output = new float[N];
    cudaMemcpy(output, dev_output, N * sizeof(float), cudaMemcpyDeviceToHost);


    // output to file
    ofstream outfile;
    outfile.open("../data/kde2d_gaussian2d.csv");

    for(int i = 0; i < N; ++i){
        outfile << input_x[i] << "," << input_y[i] << "," << output[i];
        outfile << "\n";
    }
    return 0;
}






