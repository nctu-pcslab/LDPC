#ifndef GATE_H
#define GATE_H
//#include <iostream>
#include <vector>
#include "circuit.h"
using namespace std;
class CHECKNODE;

class BITNODE{
    public:
        int ID;
        vector<CHECKNODE*> CheckNode_G;
        vector<CHECKNODE*> CheckNode_H;
#ifdef DOUBLE
        vector<double> Lcx;
        double Lint;
#else
        vector<float> Lcx;
        float Lint;
#endif
        vector<int> Lxc_position;
        ///// CUDA variable /////
        int GroupID;
};

class CHECKNODE{
    public:
        int ID;
        vector<BITNODE*> BitNode_G;
        vector<BITNODE*> BitNode_H;
#ifdef DOUBLE
        vector<double> Lxc;
#else
        vector<float> Lxc;
#endif
        vector<int> Lcx_position;
        ///// CUDA variable /////
        int GroupID;
};
#endif
