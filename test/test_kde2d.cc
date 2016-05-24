#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "../cpu/kde2d.h"
#include "../cpu/Point2d.h"

using namespace std;

int main(){
	
    // input sample data:
    vector<Point2d> sample;
    ifstream intputfile;
    intputfile.open("../data/gaussian2d.csv");
    double x;
    double y;
    while(intputfile>>x>>y){
        Point2d temp(x,y);
        sample.push_back(temp);
    }

    kde2d Gauss(1.0, sample, "uniform");


    // apply kde2d for estimation:
    vector<Point2d> intput;
    for(int i = 0; i < 100; ++i){
        for(int j = 0; j < 100; ++j){
            Point2d temp((i-50)*3.0/50,(j-50)*4.0/50);
            intput.push_back(temp);
        }
    }

    ofstream outfile;
    outfile.open("../data/kde2d_gaussian2d.csv");
    for(int i = 0; i < intput.size(); ++i){
        outfile << intput[i].x() << ',' << intput[i].y() << ',' << Gauss.pdf(intput[i]);
        outfile << "\n";
    }
    return 0;
}






