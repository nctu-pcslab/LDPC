#ifndef DEVICE_H
#define DEVICE_H

#define UCX_BLK 2048//512//256
#define UCX_THD 256//256//256
#define UXC_BLK 2548//512//256
#define UXC_THD 256//256//256
#define POS_BLK 672//672//168
#define POS_THD 128//128//128

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
        int* h_BlockBeginUXC;
        int TotalBlockUXC;
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
        int* h_BlockBegin;
        int TotalBlock;
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
        int* d_BlockBeginUXC;
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
        int* d_BlockBegin;
};
#endif
