#ifndef DEVICE_H
#define DEVICE_H

#define UCX_BLK 69//512    //BLK1
#define UCX_THD 32//512    //THD1
#define UXC_BLK UCX_BLK 
#define UXC_THD UCX_THD
#define POS_BLK 21//340   //BLK2
#define POS_THD 32//128   //THD2 

class CUDA_DATA{
    public:
        int* h_DecodedData;
        int* d_DecodedData;
        ////// BitNode on host //////
        int* h_LcxSize;
        int* h_LcxBegin;
#ifdef DOUBLE
        double* h_Lcx;
        double* h_Lint;
#else
        float* h_Lcx;
        float* h_Lint;
#endif
        int* h_NextLcxIndex;     //v2
        int* h_LxcPosition;
        ////// CheckNode on host //////
        int* h_LxcSize;
#ifdef DOUBLE
        double* h_Lxc;
#else
        float* h_Lxc;
#endif
        int* h_LxcBegin;
        int* h_NextLxcIndex;   //v2
        int* h_LcxPosition;
        ////// BitNode on device //////
        int* d_LcxSize;
        int* d_LcxBegin;
#ifdef DOUBLE
        double* d_Lcx;
        double* d_Lint;
#else
        float* d_Lcx;
        float* d_Lint;
#endif
        int* d_NextLcxIndex;
        int* d_LxcPosition;        
        ////// CheckNode on device //////
        int* d_LxcSize;
#ifdef DOUBLE
        double* d_Lxc;
#else
        float* d_Lxc;
#endif
        int* d_LxcBegin;
        int* d_NextLxcIndex;
        int* d_LcxPosition;
};
#endif
