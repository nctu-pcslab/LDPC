#ifndef DEVICE_H
#define DEVICE_H

#define UCX_BLK 2592//448//2//7//28//112//336//20
#define UCX_THD 192//128//32
#define UXC_BLK 5184//896//4//14//56//224//672//21
#define UXC_THD 192//128//32
//#define BLK3 21//21//21
//#define THD3 64//64//32

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
        int* h_LxcPosition;
        int* h_LcxStripe;       //v3.3
        ////// CheckNode on host //////
        int* h_LxcSize;
#ifdef DOUBLE
        double* h_Lxc;
#else
        float* h_Lxc;
#endif
        int* h_LxcBegin;
        int* h_LcxPosition;
        int* h_LxcStripe;       //v3.3
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
        int* d_LxcPosition;       
        int* d_LcxStripe;       //v3.3
        ////// CheckNode on device //////
        int* d_LxcSize;
#ifdef DOUBLE
        double* d_Lxc;
#else
        float* d_Lxc;
#endif
        int* d_LxcBegin;
        int* d_LcxPosition;
        int* d_LxcStripe;       //v3.3
};
#endif
