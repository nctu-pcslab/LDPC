#ifndef CIRCUIT_H
#define CIRCUIT_H
#include <vector>
#include "gauss.h"
#include "gate.h"
#include "parser.h"
#include "device.h"

//#define m 336
//#define n 672
//#define k 1/2
//#define ITERATION 10
#define SNR_Size 1          //1~13
//#define PatternNum 2978   //Total MessageData = m * PatternNum

class CIRCUIT{
    public:
        int m;
        int n;  //The size of G matrix is m*n
        int k;  //The size of H matrix is k*n
        int ITERATION;
        int PatternNum; //Total MessageData = m * NumOfCodeWord * PatternNum
        int NumOfCodeWord;
        int m_Total;
        int n_Total;
        int k_Total;
        GAUSSNUM GaussNum;
        int* MessageData;    // size = m * NumOfCodeWord = m_Total
#ifdef DOUBLE
        double* EncodedData; // size = n * NumOfCodeWord = n_Total
        double* ReceivedData;// size = n * NumOfCodeWord = n_Total
        double ErrorNum;
        double SNR[SNR_Size];
        double Variance;
#else
        float* EncodedData; // size = n * NumOfCodeWord = n_Total
        float* ReceivedData;// size = n * NumOfCodeWord = n_Total
        float ErrorNum;
        float SNR[SNR_Size];
        float Variance;
#endif
        int* DecodedData;    // size = n * NumOfCodeWord = n_Total
        vector<BITNODE*> BitNode;
        vector<CHECKNODE*> CheckNode;
        int SNR_Index;
        int Syndrome;
        TIMER Timer;
        ///// CUDA variable declaration /////
        CUDA_DATA* CudaData;
        int TotalEdge;
        ///// function declaration /////
        CIRCUIT():ErrorNum(0),Variance(0),Syndrome(1){
#ifdef DOUBLE
            double temp=1;
#else
            float temp=1;
#endif
            for(int i=0; i<SNR_Size; i++){
                SNR[i]=temp;
                temp=temp+0.2;
            }
            //SNR={1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3, 3.2, 3.4};
        }
        void Parser(char** argv);
        void ResetValue();
        void Channel_GaussianGen();
        void GenerateMessage();
        void EncodeMessage();
        void AWGN_Channel();
        void Reset_Lxc_Lcx();
        void DecodeMessage();
        void UpdateLcx();
        void UpdateLxc();
        void Calculate_Posterior();
        void Check_Syndrome();
        void Calculate_Error();
        ///// function for GPU /////
        void CUDA_MemoryAllocate();
        void CUDA_CreateDataArray();
        void CUDA_CreateDataArray2();
        void UpdateLcx_CPU();
        void UpdateLxc_CPU();
        void Calculate_Posterior_CPU();
        void MemoryCopy_H2D();
        void MemoryCopy_H2D2();
        void MemoryCopy_D2H();
        void Debug();
};
#ifdef DOUBLE
__global__ void UpdateLcx_GPU(double* d_Lxc, int* d_NextLxcIndex, int* d_LcxPosition, double* d_Lcx, int TotalEdge);
__global__ void UpdateLxc_GPU(double* d_Lcx, int* d_NextLcxIndex, int* d_LxcPosition, double* d_Lxc, double* d_Lint, int TotalEdge);
__global__ void Calculate_Posterior_GPU(int* d_LcxSize, double* d_Lcx, int* d_LcxBegin, double* d_Lint, int* d_DecodedData, int n_Total);
#else
__global__ void UpdateLcx_GPU(float* d_Lxc, int* d_NextLxcIndex, int* d_LcxPosition, float* d_Lcx, int TotalEdge);
__global__ void UpdateLxc_GPU(float* d_Lcx, int* d_NextLcxIndex, int* d_LxcPosition, float* d_Lxc, float* d_Lint, int TotalEdge);
__global__ void Calculate_Posterior_GPU(int* d_LcxSize, float* d_Lcx, int* d_LcxBegin, float* d_Lint, int* d_DecodedData, int n_Total);
#endif
#endif
