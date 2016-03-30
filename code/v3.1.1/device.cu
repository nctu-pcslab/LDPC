#include <iostream>
#include <math.h>
#include "circuit.h"
#include "device.h"
using namespace std;
//extern CIRCUIT Circuit;

void CIRCUIT::CUDA_MemoryAllocate(){
#ifdef PROFILE
    Timer.TimerStart();
#endif
    CudaData = new CUDA_DATA;
    CudaData->h_DecodedData = new int[n_Total];       //redundant, for CPU debug
    cudaMalloc((void**)&CudaData->d_DecodedData, n_Total * sizeof(int));
    ////// BitNode on host //////
    CudaData->h_LcxSize = new int [n_Total];
    CudaData->h_LcxBegin = new int [n_Total];
#ifdef DOUBLE
    CudaData->h_Lcx = new double [TotalEdge];
    CudaData->h_Lint = new double [n_Total];
#else
    CudaData->h_Lcx = new float [TotalEdge];
    CudaData->h_Lint = new float [n_Total];
#endif
    CudaData->h_LxcPosition = new int [TotalEdge];
    ////// CheckNode on host //////
    CudaData->h_LxcSize = new int [k_Total];
#ifdef DOUBLE
    CudaData->h_Lxc = new double [TotalEdge];
#else
    CudaData->h_Lxc = new float [TotalEdge];
#endif
    CudaData->h_LxcBegin = new int [k_Total];
    CudaData->h_LcxPosition = new int [TotalEdge];
    ////// BitNode on device //////
    cudaMalloc((void**)&CudaData->d_LcxSize, n_Total * sizeof(int));
    cudaMalloc((void**)&CudaData->d_LcxBegin, n_Total * sizeof(int));
#ifdef DOUBLE
    cudaMalloc((void**)&CudaData->d_Lcx, TotalEdge * sizeof(double));
    cudaMalloc((void**)&CudaData->d_Lint, n_Total * sizeof(double));
#else
    cudaMalloc((void**)&CudaData->d_Lcx, TotalEdge * sizeof(float));
    cudaMalloc((void**)&CudaData->d_Lint, n_Total * sizeof(float));
#endif
    cudaMalloc((void**)&CudaData->d_LxcPosition, TotalEdge * sizeof(int));
    ////// CheckNode on device //////
    cudaMalloc((void**)&CudaData->d_LxcSize, k_Total * sizeof(int));
#ifdef DOUBLE
    cudaMalloc((void**)&CudaData->d_Lxc, TotalEdge * sizeof(double));
#else
    cudaMalloc((void**)&CudaData->d_Lxc, TotalEdge * sizeof(float));
#endif
    cudaMalloc((void**)&CudaData->d_LxcBegin, k_Total * sizeof(int));
    cudaMalloc((void**)&CudaData->d_LcxPosition, TotalEdge * sizeof(int));
#ifdef PROFILE
    Timer.TimerFinish(Timer.tCUDA_MemoryAllocate);
#endif
}

void CIRCUIT::CUDA_CreateDataArray(){
#ifdef PROFILE
    Timer.TimerStart();
#endif
    int Begin=0;
    ///// BitNode /////
    for(unsigned i=0; i<n_Total; i++){
        CudaData->h_LcxSize[i] = BitNode[i]->Lcx.size();
        CudaData->h_LcxBegin[i] = Begin;
        //BitNode[i]->h_LcxBegin = Begin;
        Begin=Begin+BitNode[i]->Lcx.size();
        //cout<< BitNode[i]->ID<<endl;
    }
    ///// CheckNode //////
    Begin=0;
    for(unsigned i=0; i<k_Total; i++){
        CudaData->h_LxcSize[i] = CheckNode[i]->Lxc.size();
        CudaData->h_LxcBegin[i] = Begin;
        for(unsigned j=0; j<CheckNode[i]->Lxc.size(); j++){
            CudaData->h_LcxPosition[Begin+j] = CudaData->h_LcxBegin[CheckNode[i]->BitNode_H[j]->ID] + CheckNode[i]->Lcx_position[j];
        }
        Begin=Begin+CheckNode[i]->Lxc.size();
    }
    ///// BitNode /////
    for(unsigned i=0; i<n_Total; i++){
        for(unsigned j=0; j<BitNode[i]->Lcx.size(); j++)
            CudaData->h_LxcPosition[CudaData->h_LcxBegin[i]+j] = CudaData->h_LxcBegin[BitNode[i]->CheckNode_H[j]->ID] + BitNode[i]->Lxc_position[j]; 
    }
#ifdef PROFILE
    Timer.TimerFinish(Timer.tCUDA_CreateDataArray);
#endif
}

void CIRCUIT::CUDA_CreateDataArray2(){
#ifdef PROFILE
    Timer.TimerStart();
#endif

    for(unsigned i=0; i<n_Total; i++){
        CudaData->h_Lint[i] = BitNode[i]->Lint;
        /*for(unsigned j=0; j<BitNode[i]->Lcx.size(); j++){
            CudaData->h_Lcx[CudaData->h_LcxBegin[i]+j] = BitNode[i]->Lcx[j];
        }*/
    }
    for(unsigned i=0; i<k_Total; i++)
        for(unsigned j=0; j<CheckNode[i]->Lxc.size(); j++)
            CudaData->h_Lxc[CudaData->h_LxcBegin[i]+j] = CheckNode[i]->Lxc[j];

#ifdef PROFILE
    Timer.TimerFinish(Timer.tCUDA_CreateDataArray2);
#endif
}

void CIRCUIT::UpdateLcx_CPU(){
    for(int i=0; i<k_Total; i++){
        for(int j=0; j<CudaData->h_LxcSize[i]; j++){
            double sgn=1;
            double minLxc=1000;
            for(int jj=0; jj<CudaData->h_LxcSize[i]; jj++){
                if(j != jj){
                    if(CudaData->h_Lxc[CudaData->h_LxcBegin[i]+jj] > 0)
                        sgn = sgn*1;
                    else
                        sgn = sgn*(-1);
                    minLxc = min(minLxc, fabs(CudaData->h_Lxc[CudaData->h_LxcBegin[i]+jj]));
                }
            }
            CudaData->h_Lcx[CudaData->h_LcxPosition[CudaData->h_LxcBegin[i]+j]] = sgn * minLxc;
        }
    }
}

void CIRCUIT::UpdateLxc_CPU(){
    for(int i=0; i<n_Total; i++){
        double sumLcx=0;
        for(int j=0; j<CudaData->h_LcxSize[i]; j++){
            sumLcx = sumLcx + CudaData->h_Lcx[CudaData->h_LcxBegin[i]+j];
        }
        for(int j=0; j<CudaData->h_LcxSize[i]; j++){
             CudaData->h_Lxc[CudaData->h_LxcPosition[CudaData->h_LcxBegin[i]+j]] = CudaData->h_Lint[i] + (sumLcx - CudaData->h_Lcx[CudaData->h_LcxBegin[i]+j]);
        }
    }
}

void CIRCUIT::Calculate_Posterior_CPU(){
    for(int i=0; i<n_Total; i++){
        double sumLcx=0;
        for(int j=0; j<CudaData->h_LcxSize[i]; j++)
            sumLcx = sumLcx + CudaData->h_Lcx[CudaData->h_LcxBegin[i]+j];
        if(CudaData->h_Lint[i] + sumLcx >= 0)
            CudaData->h_DecodedData[i] = 0;
        else
            CudaData->h_DecodedData[i] = 1;
    }
}

void CIRCUIT::MemoryCopy_H2D(){
#ifdef PROFILE
    Timer.TimerStart();
#endif
    ////// BitNode on device //////
    cudaMemcpy(CudaData->d_LcxSize, CudaData->h_LcxSize, n_Total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_LcxBegin, CudaData->h_LcxBegin, n_Total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_LxcPosition, CudaData->h_LxcPosition, TotalEdge * sizeof(int), cudaMemcpyHostToDevice);
    ////// CheckNode on device //////
    cudaMemcpy(CudaData->d_LxcSize, CudaData->h_LxcSize, k_Total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_LxcBegin, CudaData->h_LxcBegin, k_Total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_LcxPosition, CudaData->h_LcxPosition, TotalEdge * sizeof(int), cudaMemcpyHostToDevice);

#ifdef PROFILE
    Timer.TimerFinish(Timer.tMemoryCopy_H2D);
#endif
}

void CIRCUIT::MemoryCopy_H2D2(){
#ifdef PROFILE
    Timer.TimerStart();
#endif
    ////// BitNode on device //////
#ifdef DOUBLE
    //cudaMemcpy(CudaData->d_Lcx, CudaData->h_Lcx, TotalEdge * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_Lint, CudaData->h_Lint, n_Total * sizeof(double), cudaMemcpyHostToDevice);
    ////// CheckNode on device //////
    cudaMemcpy(CudaData->d_Lxc, CudaData->h_Lxc, TotalEdge * sizeof(double), cudaMemcpyHostToDevice);
#else
    cudaMemcpy(CudaData->d_Lint, CudaData->h_Lint, n_Total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_Lxc, CudaData->h_Lxc, TotalEdge * sizeof(float), cudaMemcpyHostToDevice);
#endif

#ifdef PROFILE
    Timer.TimerFinish(Timer.tMemoryCopy_H2D2);
#endif
}

void CIRCUIT::MemoryCopy_D2H(){
#ifdef PROFILE
    Timer.TimerStart();
#endif
    //cudaMemcpy(CudaData->h_Lcx, CudaData->d_Lcx, TotalEdge * sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(CudaData->h_DecodedData, CudaData->d_DecodedData, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(DecodedData, CudaData->d_DecodedData, n_Total * sizeof(int), cudaMemcpyDeviceToHost);
#ifdef PROFILE
    Timer.TimerFinish(Timer.tMemoryCopy_D2H);
#endif
}
#ifdef DOUBLE
__global__ void UpdateLcx_GPU(int* d_LxcSize, double* d_Lxc, int* d_LxcBegin, int* d_LcxPosition, double* d_Lcx,int k_Total){
#else
__global__ void UpdateLcx_GPU(int* d_LxcSize, float* d_Lxc, int* d_LxcBegin, int* d_LcxPosition, float* d_Lcx,int k_Total){
#endif
    int total_task = gridDim.x * blockDim.x;
    int task_sn = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=task_sn; i<k_Total; i+=total_task){
        for(int j=0; j<d_LxcSize[i]; j++){
#ifdef DOUBLE
            double sgn=1;
            double minLxc=1000;
#else
            float sgn=1;
            float minLxc=1000;
#endif
            for(int jj=0; jj<d_LxcSize[i]; jj++){
                if(j != jj){
                    if(d_Lxc[d_LxcBegin[i]+jj] > 0)
                        sgn = sgn*1;
                    else
                        sgn = sgn*(-1);
                    minLxc = min(minLxc, fabs(d_Lxc[d_LxcBegin[i]+jj]));
                }
            }
            d_Lcx[d_LcxPosition[d_LxcBegin[i]+j]] = sgn * minLxc;
        }
    }
}
#ifdef DOUBLE
__global__ void UpdateLxc_GPU(int* d_LcxSize, double* d_Lcx, int* d_LcxBegin, int* d_LxcPosition, double* d_Lxc, double* d_Lint, int n_Total){
#else
__global__ void UpdateLxc_GPU(int* d_LcxSize, float* d_Lcx, int* d_LcxBegin, int* d_LxcPosition, float* d_Lxc, float* d_Lint, int n_Total){
#endif
    int total_task = gridDim.x * blockDim.x;
    int task_sn = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=task_sn; i<n_Total; i+=total_task){
#ifdef DOUBLE
        double sumLcx=0;
#else
        float sumLcx=0;
#endif
        for(int j=0; j<d_LcxSize[i]; j++){
            sumLcx = sumLcx + d_Lcx[d_LcxBegin[i]+j];
        }
        for(int j=0; j<d_LcxSize[i]; j++){
             d_Lxc[d_LxcPosition[d_LcxBegin[i]+j]] = d_Lint[i] + (sumLcx - d_Lcx[d_LcxBegin[i]+j]);
        }
    }
}
#ifdef DOUBLE
__global__ void Calculate_Posterior_GPU(int* d_LcxSize, double* d_Lcx, int* d_LcxBegin, double* d_Lint, int* d_DecodedData,int n_Total){
#else
__global__ void Calculate_Posterior_GPU(int* d_LcxSize, float* d_Lcx, int* d_LcxBegin, float* d_Lint, int* d_DecodedData,int n_Total){
#endif
    int total_task = gridDim.x * blockDim.x;
    int task_sn = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=task_sn; i<n_Total; i+=total_task){
#ifdef DOUBLE
        double sumLcx=0;
#else
        float sumLcx=0;
#endif
        for(int j=0; j<d_LcxSize[i]; j++)
            sumLcx = sumLcx + d_Lcx[d_LcxBegin[i]+j];
        if(d_Lint[i] + sumLcx >= 0)
            d_DecodedData[i] = 0;
        else
            d_DecodedData[i] = 1;
    }
}

void CIRCUIT::Debug(){
    /*for(int i=0; i<m; i++){
        for(unsigned j=0; j<CheckNode[i]->Lxc.size(); j++){
            if(CheckNode[i]->Lxc[j] != CudaData->h_Lxc[CudaData->h_LxcBegin[i]+j])
                cout<<CheckNode[i]->Lxc[j]<<"   "<<CudaData->h_Lxc[CudaData->h_LxcBegin[i]+j]<<endl;
        }
    }
    for(int i=0; i<n; i++){
        for(unsigned j=0; j<BitNode[i]->Lcx.size(); j++){
            if(BitNode[i]->Lcx[j] != CudaData->h_Lcx[CudaData->h_LcxBegin[i]+j])
                cout<<BitNode[i]->Lcx[j]<<" "<<CudaData->h_Lcx[CudaData->h_LcxBegin[i]+j]<<endl;
        }
    }*/
    //cout<<"stop"<<endl;
    //getchar();
    for(int i=0; i<n; i++){
        if(DecodedData[i] != CudaData->h_DecodedData[i])
            cout<<DecodedData[i]<<" "<<CudaData->h_DecodedData[i]<<endl;
    }
}
