#include <iostream>
#include <math.h>
#include <algorithm>
#include "circuit.h"

using namespace std;

void CIRCUIT::Channel_GaussianGen(){
    double t1, t2, w;
    //srand(time(NULL));
    do{
        t1 = 2.0*((double)rand()/RAND_MAX) - 1.0;
        t2 = 2.0*((double)rand()/RAND_MAX) - 1.0;
        w = t1*t1 + t2*t2;
    }while(w>=1.0);

    w = sqrt((-2.0*log(w))/w);

    GaussNum.x1 = t1*w;
    GaussNum.x2 = t2*w;
}
