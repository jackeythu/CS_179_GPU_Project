#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <exception>
#include "../cpu/kde1d.h"

using namespace std;

void PrintVector(vector<double> x){
    cout << "{" << endl;
    for(int i = 0; i < x.size(); ++i){
        cout << x[i] << ",";
    }
    cout << "}" << endl;
    return;
}

int main(int argc, char* argv[]){

	// Input sample data, kernel_type, naive/adaptive
    if(argc != 4){
        printf("Usage: <sample_data_path> <kernel_type:gaussian/uniform/quartic> <naive/adaptive>\n");
        exit(-1);
    }

    std::string path(argv[1]);
    std::string kernel_type(argv[2]);
    std::string version_type(argv[3]);
    bool version;
    if(version_type == "naive"){
        version = false;
    }
    else version = true;

    // input sample data:

	ifstream myfile;
	myfile.open(path);
	myfile.is_open() ? cout << "open data file successfully" : cout << "Error: Cannot open data file!";
	cout << endl;

	vector<double> sample;
    float x = 0;
    while(myfile >> x){
        sample.push_back(x);
    }
    cout << "finish inputting data" << endl;


    // apply kde1d for estimation:

    double bandwidth = 0.8;
    kde1d Gauss(bandwidth, sample, "gaussian", version);

    int N = 10000;
    vector<double> input;
    for(int i = 0; i < N; ++i){
    	input.push_back((i - N/2) * 3.0 / (N/2));
    }


    float time_initial, time_final;
    time_initial = clock();

    vector<double> result;
    for(int i = 0; i < input.size(); ++i){
    	result.push_back(Gauss.pdf(input[i]));
    }
    time_final = clock();

    
    // output result:

    cout << "cost time is " << (time_final - time_initial)/CLOCKS_PER_SEC << endl;
    
    ofstream outfile;
    outfile.open("../data/kde1d_gauss1d_single.csv");
    for(int i = 0; i < result.size(); ++i){
        outfile << input[i] << ',' << result[i];
        outfile << "\n";
    }

    cout << "writing estimation to file Done!" << endl;
    return 0;
}






