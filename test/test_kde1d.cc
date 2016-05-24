#include <vector>
#include <iostream>
#include <fstream>
#include <string>
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

int main(){
	
    // input sample data:

	ifstream myfile;
	myfile.open("../data/gauss1d_single.csv");
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
    kde1d Gauss(bandwidth, sample, "gaussian", true);

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






