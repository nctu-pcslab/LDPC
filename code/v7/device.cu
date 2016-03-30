#include <iostream>
#include <math.h>
#include <cstdio>
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
    CudaData->h_LcxSize = new int [n_Total];              //redundant?
    CudaData->h_LcxBegin = new int [n_Total];        
#ifdef DOUBLE
    CudaData->h_Lcx = new double [TotalEdge];
    CudaData->h_Lint = new double [TotalEdge];      //v2
#else
    CudaData->h_Lcx = new float [TotalEdge];
    CudaData->h_Lint = new float [TotalEdge];
#endif
    CudaData->h_NextLcxIndex = new int [TotalEdge];  //v2
    CudaData->h_LxcPosition = new int [TotalEdge];
    ////// CheckNode on host //////
    CudaData->h_LxcSize = new int [k_Total];              //redundant?
#ifdef DOUBLE
    CudaData->h_Lxc = new double [TotalEdge];
#else
    CudaData->h_Lxc = new float [TotalEdge];
#endif
    CudaData->h_LxcBegin = new int [k_Total];
    CudaData->h_NextLxcIndex = new int [TotalEdge];  //v2
    CudaData->h_LcxPosition = new int [TotalEdge];
    ////// BitNode on device //////
    cudaMalloc((void**)&CudaData->d_LcxSize, n_Total * sizeof(int));  //redundant?
    cudaMalloc((void**)&CudaData->d_LcxBegin, n_Total * sizeof(int)); //redundant?
#ifdef DOUBLE
    cudaMalloc((void**)&CudaData->d_Lcx, TotalEdge * sizeof(double));
    cudaMalloc((void**)&CudaData->d_Lint, TotalEdge * sizeof(double));
#else
    cudaMalloc((void**)&CudaData->d_Lcx, TotalEdge * sizeof(float));
    cudaMalloc((void**)&CudaData->d_Lint, TotalEdge * sizeof(float));
#endif
    cudaMalloc((void**)&CudaData->d_NextLcxIndex, TotalEdge * sizeof(int));
    cudaMalloc((void**)&CudaData->d_LxcPosition, TotalEdge * sizeof(int));
    ////// CheckNode on device //////
    cudaMalloc((void**)&CudaData->d_LxcSize, k_Total * sizeof(int));  //redundant?
#ifdef DOUBLE
    cudaMalloc((void**)&CudaData->d_Lxc, TotalEdge * sizeof(double));
#else
    cudaMalloc((void**)&CudaData->d_Lxc, TotalEdge * sizeof(float));
#endif
    cudaMalloc((void**)&CudaData->d_LxcBegin, k_Total * sizeof(int)); //redundant?
    cudaMalloc((void**)&CudaData->d_NextLxcIndex, TotalEdge * sizeof(int));
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
    int Index=0;    //v2
    ///// BitNode /////
    for(unsigned i=0; i<n_Total; i++){
        CudaData->h_LcxSize[i] = BitNode[i]->Lcx.size();
        CudaData->h_LcxBegin[i] = Begin;
        for(unsigned j=0; j<BitNode[i]->Lcx.size(); j++){   //v2
            CudaData->h_NextLcxIndex[Index] = Index+1;
            Index++;
        }
        CudaData->h_NextLcxIndex[Index-1] = Begin;
        Begin=Begin+BitNode[i]->Lcx.size();
    }
    ///// CheckNode //////
    Begin=0;
    Index=0;    //v2
    for(unsigned i=0; i<k_Total; i++){
        CudaData->h_LxcSize[i] = CheckNode[i]->Lxc.size();
        CudaData->h_LxcBegin[i] = Begin;
        for(unsigned j=0; j<CheckNode[i]->Lxc.size(); j++){
            CudaData->h_LcxPosition[Begin+j] = CudaData->h_LcxBegin[CheckNode[i]->BitNode_H[j]->ID] + CheckNode[i]->Lcx_position[j];
            CudaData->h_NextLxcIndex[Index] = Index + 1;
            Index++;
        }
        CudaData->h_NextLxcIndex[Index-1] = Begin;
        Begin=Begin+CheckNode[i]->Lxc.size();
    }
    ///// BitNode /////
    for(unsigned i=0; i<n_Total; i++){
        for(unsigned j=0; j<BitNode[i]->Lcx.size(); j++)
            CudaData->h_LxcPosition[CudaData->h_LcxBegin[i]+j] = CudaData->h_LxcBegin[BitNode[i]->CheckNode_H[j]->ID] + BitNode[i]->Lxc_position[j]; 
    }
    ///// Debug /////
    //Index = 0;
    /*for(unsigned i=0; i<n; i++){
        for(unsigned j=0; j<BitNode[i]->Lcx.size(); j++){   //v2
            cout<<CudaData->h_NextLcxIndex[Index]<<" ";
            Index++;
        }
        cout<<endl;
    }*/
    /*for(unsigned i=0; i<m; i++){
        for(unsigned j=0; j<CheckNode[i]->Lxc.size(); j++){
            cout<<CudaData->h_NextLxcIndex[Index]<<" ";
            Index++;
        }
        cout<<endl;
    }
    cout<<"TotalEdge = "<<TotalEdge<<endl;
    char a;
    cin>>a;*/
    //////// UpdateLcx_GPU /////////////
    CudaData->h_BlockBegin = new int[k_Total];
    //cudaMalloc((void**)&CudaData->d_BlockBegin, (k_Total) * sizeof(int));
    int count=0;
    int count2=0;
    CudaData->TotalBlock = 1;
    CudaData->h_BlockBegin[0]=0;
    for(unsigned i=0; i<k_Total; i++){
        count=count+CheckNode[i]->Lxc.size();
        count2=count2+CheckNode[i]->Lxc.size();
        if(count > UCX_THD){
            CudaData->h_BlockBegin[CudaData->TotalBlock] = count2 - CheckNode[i]->Lxc.size();
            count = CheckNode[i]->Lxc.size();
            CudaData->TotalBlock++;
        }
    }
    CudaData->h_BlockBegin[CudaData->TotalBlock] = count2;
    cudaMalloc((void**)&CudaData->d_BlockBegin, (CudaData->TotalBlock+1) * sizeof(int));
    cudaMemcpy(CudaData->d_BlockBegin, CudaData->h_BlockBegin, (CudaData->TotalBlock+1) * sizeof(int), cudaMemcpyHostToDevice);
    ///////// debug /////////
    /*for(unsigned i=0; i<CudaData->TotalBlock-1; i++)
        cout<<"Begin:"<<CudaData->h_BlockBegin[i]<<"  Size:"<<CudaData->h_BlockBegin[i+1]-CudaData->h_BlockBegin[i]<<endl;
    cout<<"Begin:"<<CudaData->h_BlockBegin[CudaData->TotalBlock-1]<<"  Size:"<<CudaData->h_BlockBegin[CudaData->TotalBlock]-CudaData->h_BlockBegin[CudaData->TotalBlock-1]<<endl;
    for(unsigned i=0; i<CudaData->TotalBlock+1; i++)
        cout<<CudaData->h_BlockBegin[i]<<endl;
    getchar();*/
    //////// UpdateLxc_GPU /////////////
    CudaData->h_BlockBeginUXC = new int[n_Total];
    count=0;
    count2=0;
    CudaData->TotalBlockUXC = 1;
    CudaData->h_BlockBeginUXC[0]=0;
    for(unsigned i=0; i<n_Total; i++){
        count=count+BitNode[i]->Lcx.size();
        count2=count2+BitNode[i]->Lcx.size();
        if(count > UXC_THD){
            CudaData->h_BlockBeginUXC[CudaData->TotalBlockUXC] = count2 - BitNode[i]->Lcx.size();
            count = BitNode[i]->Lcx.size();
            CudaData->TotalBlockUXC++;
        }
    }
    CudaData->h_BlockBeginUXC[CudaData->TotalBlockUXC] = count2;
    cudaMalloc((void**)&CudaData->d_BlockBeginUXC, (CudaData->TotalBlockUXC+1) * sizeof(int));
    cudaMemcpy(CudaData->d_BlockBeginUXC, CudaData->h_BlockBeginUXC, (CudaData->TotalBlockUXC+1) * sizeof(int), cudaMemcpyHostToDevice);
    ///////// debug /////////
    /*for(unsigned i=0; i<CudaData->TotalBlockUXC-1; i++)
        cout<<"Begin:"<<CudaData->h_BlockBeginUXC[i]<<"  Size:"<<CudaData->h_BlockBeginUXC[i+1]-CudaData->h_BlockBeginUXC[i]<<endl;
    cout<<"Begin:"<<CudaData->h_BlockBeginUXC[CudaData->TotalBlockUXC-1]<<"  Size:"<<CudaData->h_BlockBeginUXC[CudaData->TotalBlockUXC]-CudaData->h_BlockBeginUXC[CudaData->TotalBlockUXC-1]<<endl;
    for(unsigned i=0; i<CudaData->TotalBlockUXC+1; i++)
        cout<<CudaData->h_BlockBeginUXC[i]<<endl;
    getchar();*/
#ifdef PROFILE
    Timer.TimerFinish(Timer.tCUDA_CreateDataArray);
#endif
}

void CIRCUIT::CUDA_CreateDataArray2(){
#ifdef PROFILE
    Timer.TimerStart();
#endif

    for(unsigned i=0; i<n_Total; i++){
        //CudaData->h_Lint[i] = BitNode[i]->Lint;
        for(unsigned j=0; j<BitNode[i]->Lcx.size(); j++){
            //CudaData->h_Lcx[CudaData->h_LcxBegin[i]+j] = BitNode[i]->Lcx[j];
            CudaData->h_Lint[CudaData->h_LcxBegin[i]+j] = BitNode[i]->Lint;
        }
    }
    for(unsigned i=0; i<k_Total; i++)
        for(unsigned j=0; j<CheckNode[i]->Lxc.size(); j++)
            CudaData->h_Lxc[CudaData->h_LxcBegin[i]+j] = CheckNode[i]->Lxc[j];

#ifdef PROFILE
    Timer.TimerFinish(Timer.tCUDA_CreateDataArray2);
#endif
}

void CIRCUIT::UpdateLcx_CPU(){
    for(int i=0; i<TotalEdge; i++){
        double sgn=1;
        double minLxc=1000;
        int Index = CudaData->h_NextLxcIndex[i];
        while(Index != i){
            if(CudaData->h_Lxc[Index] > 0)
                sgn = sgn*1;
            else
                sgn = sgn*(-1);
            minLxc = min(minLxc, fabs(CudaData->h_Lxc[Index]));
            Index = CudaData->h_NextLxcIndex[Index];
        }
        CudaData->h_Lcx[CudaData->h_LcxPosition[i]] = sgn * minLxc;        
    }
}

void CIRCUIT::UpdateLxc_CPU(){
    for(int i=0; i<TotalEdge; i++){
        double sumLcx=0;
        int Index = CudaData->h_NextLcxIndex[i];
        while(Index != i){
            sumLcx = sumLcx + CudaData->h_Lcx[Index];
            Index = CudaData->h_NextLcxIndex[Index];
        }
        CudaData->h_Lxc[CudaData->h_LxcPosition[i]] = CudaData->h_Lint[i] + sumLcx;
    }
}

void CIRCUIT::Calculate_Posterior_CPU(){
    for(int i=0; i<n_Total; i++){
        double sumLcx=0;
        for(int j=0; j<CudaData->h_LcxSize[i]; j++)
            sumLcx = sumLcx + CudaData->h_Lcx[CudaData->h_LcxBegin[i]+j];
        if(CudaData->h_Lint[CudaData->h_LcxBegin[i]] + sumLcx >= 0)
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
    cudaMemcpy(CudaData->d_LcxSize, CudaData->h_LcxSize, n_Total * sizeof(int), cudaMemcpyHostToDevice);//redundant
    cudaMemcpy(CudaData->d_LcxBegin, CudaData->h_LcxBegin, n_Total * sizeof(int), cudaMemcpyHostToDevice);//redundant
    cudaMemcpy(CudaData->d_NextLcxIndex, CudaData->h_NextLcxIndex, TotalEdge * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_LxcPosition, CudaData->h_LxcPosition, TotalEdge * sizeof(int), cudaMemcpyHostToDevice);
    ////// CheckNode on device //////
    cudaMemcpy(CudaData->d_LxcSize, CudaData->h_LxcSize, k_Total * sizeof(int), cudaMemcpyHostToDevice);//redundant
    cudaMemcpy(CudaData->d_LxcBegin, CudaData->h_LxcBegin, k_Total * sizeof(int), cudaMemcpyHostToDevice);//redundant
    cudaMemcpy(CudaData->d_NextLxcIndex, CudaData->h_NextLxcIndex, TotalEdge * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_LcxPosition, CudaData->h_LcxPosition, TotalEdge * sizeof(int), cudaMemcpyHostToDevice);

#ifdef PROFILE
    Timer.TimerFinish(Timer.tMemoryCopy_H2D);
#endif
}

void CIRCUIT::MemoryCopy_H2D2(){
#ifdef PROFILE
    Timer.TimerStart();
#endif
#ifdef DOUBLE
    ////// BitNode on device //////
    //cudaMemcpy(CudaData->d_Lcx, CudaData->h_Lcx, TotalEdge * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_Lint, CudaData->h_Lint, TotalEdge * sizeof(double), cudaMemcpyHostToDevice);
    ////// CheckNode on device //////
    cudaMemcpy(CudaData->d_Lxc, CudaData->h_Lxc, TotalEdge * sizeof(double), cudaMemcpyHostToDevice);
#else
    cudaMemcpy(CudaData->d_Lint, CudaData->h_Lint, TotalEdge * sizeof(float), cudaMemcpyHostToDevice);
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
__global__ void UpdateLcx_GPU(double* d_Lxc, int* d_NextLxcIndex, int* d_LcxPosition, double* d_Lcx, int TotalEdge, int* d_BlockBegin, int TotalBlock){
#else
__global__ void UpdateLcx_GPU(float* d_Lxc, int* d_NextLxcIndex, int* d_LcxPosition, float* d_Lcx, int TotalEdge, int* d_BlockBegin, int TotalBlock){
#endif
#ifdef DOUBLE
    __shared__ double s_Lxc[UCX_THD];
#else
    __shared__ float s_Lxc[UCX_THD];
#endif
    __shared__ int s_NextLxcIndex[UCX_THD];
    for(int i=blockIdx.x; i<TotalBlock; i+=gridDim.x){
        if(threadIdx.x < d_BlockBegin[i+1] - d_BlockBegin[i]){
            int task_sn = d_BlockBegin[i]+threadIdx.x;
            s_Lxc[threadIdx.x] = d_Lxc[task_sn];
            s_NextLxcIndex[threadIdx.x] = d_NextLxcIndex[task_sn] - (task_sn - threadIdx.x);
            __syncthreads();
            
#ifdef DOUBLE
            double sgn=1;
            double minLxc=1000;
#else
            float sgn=1;
            float minLxc=1000;
#endif
            int Index = s_NextLxcIndex[threadIdx.x];
            while(Index != threadIdx.x){
                if(s_Lxc[Index] < 0){
                    sgn = sgn*(-1);
                }
                minLxc = min(minLxc, fabs(s_Lxc[Index]));
                Index = s_NextLxcIndex[Index];
            }
            d_Lcx[d_LcxPosition[task_sn]] = sgn * minLxc;
        }
        //__syncthreads();
    }
    /*for(int i=blockIdx.x; i<TotalBlock; i+=gridDim.x){
        if(threadIdx.x < d_BlockBegin[i+1] - d_BlockBegin[i]){
            float sgn=1;
            float minLxc=1000;
            int task_sn = d_BlockBegin[i]+threadIdx.x;
            int Index = d_NextLxcIndex[task_sn];
            while(Index != task_sn){
                if(d_Lxc[Index] < 0)
                    sgn = sgn*(-1);
                minLxc = min(minLxc, fabs(d_Lxc[Index]));
                Index = d_NextLxcIndex[Index];        
            }
            d_Lcx[d_LcxPosition[task_sn]] = sgn * minLxc;
        }
    }*/
}
#ifdef DOUBLE
__global__ void UpdateLxc_GPU(double* d_Lcx, int* d_NextLcxIndex, int* d_LxcPosition, double* d_Lxc, double* d_Lint, int TotalEdge, int* d_BlockBegin, int TotalBlock){
#else
__global__ void UpdateLxc_GPU(float* d_Lcx, int* d_NextLcxIndex, int* d_LxcPosition, float* d_Lxc, float* d_Lint, int TotalEdge, int* d_BlockBegin, int TotalBlock){
#endif
/*    int total_task = gridDim.x * blockDim.x;
    int task_sn = blockIdx.x*blockDim.x + threadIdx.x;

    for(int i=task_sn; i<TotalEdge; i+=total_task){
#ifdef DOUBLE
        double sumLcx=0;
#else
        float sumLcx=0;
#endif
        int Index = d_NextLcxIndex[i];
        while(Index != i){
            sumLcx = sumLcx + d_Lcx[Index];
            Index = d_NextLcxIndex[Index];
        }
        d_Lxc[d_LxcPosition[i]] = d_Lint[i] + sumLcx;
    }*/

    __shared__ float s_Lcx[UXC_THD];
    __shared__ int s_NextLcxIndex[UXC_THD];

    /*for(int i=blockIdx.x; i<TotalBlock; i+=gridDim.x){
        if(threadIdx.x < d_BlockBegin[i+1] - d_BlockBegin[i]){
            int task_sn = d_BlockBegin[i]+threadIdx.x;
            float sumLcx=0;
            s_Lcx[threadIdx.x] = d_Lcx[task_sn];
            s_NextLcxIndex[threadIdx.x] = d_NextLcxIndex[task_sn];
            __syncthreads();
            
            int Index = s_NextLcxIndex[threadIdx.x];
            while(Index != task_sn){
                sumLcx = sumLcx + s_Lcx[Index - task_sn + threadIdx.x];
                Index = s_NextLcxIndex[Index - task_sn + threadIdx.x];
            }
            d_Lxc[d_LxcPosition[task_sn]] = d_Lint[task_sn] + sumLcx;
        }
        __syncthreads();
    }*/
    /*for(int i=blockIdx.x; i<TotalBlock; i+=gridDim.x){
        if(threadIdx.x < d_BlockBegin[i+1] - d_BlockBegin[i]){
            int task_sn = d_BlockBegin[i]+threadIdx.x;
            float sumLcx=0;
            s_Lcx[threadIdx.x] = d_Lcx[task_sn];
            s_NextLcxIndex[threadIdx.x] = d_NextLcxIndex[task_sn];
            __syncthreads();
            s_NextLcxIndex[threadIdx.x] = s_NextLcxIndex[threadIdx.x] - task_sn + threadIdx.x;
            __syncthreads();
            int Index = s_NextLcxIndex[threadIdx.x];
            //while(Index != task_sn){
            while(Index != threadIdx.x){
                sumLcx = sumLcx + s_Lcx[Index];
                Index = s_NextLcxIndex[Index];
            }
            d_Lxc[d_LxcPosition[task_sn]] = d_Lint[task_sn] + sumLcx;
        }
    }*/
    for(int i=blockIdx.x; i<TotalBlock; i+=gridDim.x){
       /* int task_sn = d_BlockBegin[i]+threadIdx.x; 
        s_Lcx[threadIdx.x] = d_Lcx[task_sn];
            s_NextLcxIndex[threadIdx.x] = d_NextLcxIndex[task_sn] - (task_sn - threadIdx.x);
            __syncthreads();*/
        if(threadIdx.x < d_BlockBegin[i+1] - d_BlockBegin[i]){
            int task_sn = d_BlockBegin[i]+threadIdx.x;
            s_Lcx[threadIdx.x] = d_Lcx[task_sn];
            s_NextLcxIndex[threadIdx.x] = d_NextLcxIndex[task_sn] - (task_sn - threadIdx.x);
            __syncthreads();
            
            float sumLcx=0;
            int Index = s_NextLcxIndex[threadIdx.x];
            while(Index != threadIdx.x){
                sumLcx = sumLcx + s_Lcx[Index];
                Index = s_NextLcxIndex[Index];
            }
            d_Lxc[d_LxcPosition[task_sn]] = d_Lint[task_sn] + sumLcx;
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
        if(d_Lint[d_LcxBegin[i]] + sumLcx >= 0)
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
    for(int i=0; i<n_Total; i++){
        if(DecodedData[i] != CudaData->h_DecodedData[i])
            cout<<DecodedData[i]<<" "<<CudaData->h_DecodedData[i]<<endl;
    }
}
