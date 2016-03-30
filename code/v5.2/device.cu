#include <iostream>
#include <cstdio>
#include <math.h>
#include "circuit.h"
#include "device.h"
#include <map>
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
    CudaData->h_LcxStripe = new int [n_Total];      //v3.3
    ////// CheckNode on host //////
    CudaData->h_LxcSize = new int [k_Total];
#ifdef DOUBLE
    CudaData->h_Lxc = new double [TotalEdge];
#else
    CudaData->h_Lxc = new float [TotalEdge];
#endif
    CudaData->h_LxcBegin = new int [k_Total];
    CudaData->h_LcxPosition = new int [TotalEdge];
    CudaData->h_LxcStripe = new int [k_Total];      //v3.3
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
    cudaMalloc((void**)&CudaData->d_LcxStripe, n_Total * sizeof(int));  //v3.3
    ////// CheckNode on device //////
    cudaMalloc((void**)&CudaData->d_LxcSize, k_Total * sizeof(int));
#ifdef DOUBLE
    cudaMalloc((void**)&CudaData->d_Lxc, TotalEdge * sizeof(double));
#else
    cudaMalloc((void**)&CudaData->d_Lxc, TotalEdge * sizeof(float));
#endif
    cudaMalloc((void**)&CudaData->d_LxcBegin, k_Total * sizeof(int));
    cudaMalloc((void**)&CudaData->d_LcxPosition, TotalEdge * sizeof(int));
    cudaMalloc((void**)&CudaData->d_LxcStripe, k_Total * sizeof(int));  //v3.3
#ifdef PROFILE
    Timer.TimerFinish(Timer.tCUDA_MemoryAllocate);
#endif
}

void CIRCUIT::CUDA_CreateDataArray(){
#ifdef PROFILE
    Timer.TimerStart();
#endif
    ///// BitNode /////
    vector<int> GroupSize;
    int temp=1;
    BitNode[0]->GroupID=0;
    for(unsigned i=1; i<n; i++){
        if(i != n-1){
            if(BitNode[i]->Lcx.size() != BitNode[i-1]->Lcx.size()){
                GroupSize.push_back(temp);
                temp=1;
                BitNode[i]->GroupID=GroupSize.size();
            }
            else{
                temp++;
                BitNode[i]->GroupID=GroupSize.size();
            }
        }
        else{
            if(BitNode[i]->Lcx.size() != BitNode[i-1]->Lcx.size()){
                GroupSize.push_back(temp);
                temp=1;
                BitNode[i]->GroupID=GroupSize.size();
                GroupSize.push_back(temp);
            }
            else{
                temp++;
                BitNode[i]->GroupID=GroupSize.size();
                GroupSize.push_back(temp);
            }
        }
    }

    //////// debug ///////
    /*for(unsigned i=0; i<GroupSize.size(); i++)
        cout<<GroupSize[i]<<endl;
    getchar();*/
    int Stripe=0;
    int PreviousNode=0;
    for(unsigned i=0; i<n_Total; i++){
        if(i>0){
            if(BitNode[i]->Lcx.size() != BitNode[i-1]->Lcx.size()){
                Stripe = Stripe + BitNode[i-1]->Lcx.size() * GroupSize[BitNode[(i-1)%n]->GroupID];
                PreviousNode = i;
            }
        }
        CudaData->h_LcxSize[i] = BitNode[i]->Lcx.size();
        CudaData->h_LcxBegin[i] = Stripe + i - PreviousNode;
        CudaData->h_LcxStripe[i] = GroupSize[BitNode[i%n]->GroupID];
    }
    ///// CheckNode //////
    GroupSize.clear();
    temp=1;
    CheckNode[0]->GroupID=0;
    for(unsigned i=1; i<k; i++){
        if(i != k-1){
            if(CheckNode[i]->Lxc.size() != CheckNode[i-1]->Lxc.size()){
                GroupSize.push_back(temp);
                temp=1;
                CheckNode[i]->GroupID=GroupSize.size();
            }
            else{
                temp++;
                CheckNode[i]->GroupID=GroupSize.size();
            }
        }
        else{
            if(CheckNode[i]->Lxc.size() != CheckNode[i-1]->Lxc.size()){
                GroupSize.push_back(temp);
                temp=1;
                CheckNode[i]->GroupID=GroupSize.size();
                GroupSize.push_back(temp);
            }
            else{
                temp++;
                CheckNode[i]->GroupID=GroupSize.size();
                GroupSize.push_back(temp);
            }
        }
    }
    //////// debug ///////
    /*for(unsigned i=0; i<GroupSize.size(); i++)
        cout<<GroupSize[i]<<endl;
    getchar();*/
    Stripe=0;
    PreviousNode=0;
    for(unsigned i=0; i<k_Total; i++){
        if(i>0){
            if(CheckNode[i]->Lxc.size() != CheckNode[i-1]->Lxc.size()){
                Stripe = Stripe + CheckNode[i-1]->Lxc.size() * GroupSize[CheckNode[(i-1)%k]->GroupID];
                PreviousNode = i;
            }
        }
        CudaData->h_LxcSize[i] = CheckNode[i]->Lxc.size();
        CudaData->h_LxcBegin[i] = Stripe + i - PreviousNode;
        CudaData->h_LxcStripe[i] = GroupSize[CheckNode[i%k]->GroupID];
        for(unsigned j=0; j<CheckNode[i]->Lxc.size(); j++)
            CudaData->h_LcxPosition[CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j] = CudaData->h_LcxBegin[CheckNode[i]->BitNode_H[j]->ID] + CudaData->h_LcxStripe[CheckNode[i]->BitNode_H[j]->ID] * CheckNode[i]->Lcx_position[j];                
    }
    ///// BitNode /////
    for(unsigned i=0; i<n_Total; i++){
        for(unsigned j=0; j<BitNode[i]->Lcx.size(); j++)
            CudaData->h_LxcPosition[CudaData->h_LcxBegin[i]+CudaData->h_LcxStripe[i]*j] = CudaData->h_LxcBegin[BitNode[i]->CheckNode_H[j]->ID] + CudaData->h_LxcStripe[BitNode[i]->CheckNode_H[j]->ID] * BitNode[i]->Lxc_position[j];
    }
    ////////// debug ///////////
    /*cout<<"h_LcxSize[] = ";
    for(unsigned i=0; i<n_Total; i++)
        cout<<CudaData->h_LcxSize[i]<<" ";
    cout<<endl;
    cout<<"h_LcxBegin[] = ";
    for(unsigned i=0; i<n_Total; i++)
        cout<<CudaData->h_LcxBegin[i]<<" ";
    cout<<endl;
    cout<<"h_LcxStripe[] = ";
    for(unsigned i=0; i<n_Total; i++)
        cout<<CudaData->h_LcxStripe[i]<<" ";
    cout<<endl;*/
    /*cout<<"h_LxcSize[] = ";
    for(unsigned i=0; i<k_Total; i++)
        cout<<CudaData->h_LxcSize[i]<<" ";
    cout<<endl;
    cout<<"h_LxcBegin[] = ";
    for(unsigned i=0; i<k_Total; i++)
        cout<<CudaData->h_LxcBegin[i]<<" ";
    cout<<endl;
    cout<<"h_LxcStripe[] = ";
    for(unsigned i=0; i<k_Total; i++)
        cout<<CudaData->h_LxcStripe[i]<<" ";
    cout<<endl;*/
    /*for(unsigned i=0; i<TotalEdge; i++)
        cout<<CudaData->h_LxcPosition[i]<<" ";
    cout<<endl;
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
        CudaData->h_Lint[i] = BitNode[i]->Lint;
        /*for(unsigned j=0; j<BitNode[i]->Lcx.size(); j++){
            CudaData->h_Lcx[CudaData->h_LcxBegin[i]+CudaData->h_LcxStripe[i]*j] = BitNode[i]->Lcx[j];
        }*/
    }
    for(unsigned i=0; i<k_Total; i++){
        for(unsigned j=0; j<CheckNode[i]->Lxc.size(); j++){
            /*if(CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j > TotalEdge){
                cout<<"Something wrong! "<<CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j<<endl;
            }*/
            CudaData->h_Lxc[CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j] = CheckNode[i]->Lxc[j];
        }
    }

#ifdef PROFILE
    Timer.TimerFinish(Timer.tCUDA_CreateDataArray2);
#endif
}

void CIRCUIT::UpdateLcx_CPU(){
    for(int i=0; i<k_Total; i++){
        double sgn=1;
        double minLxc1=1000;
        double minLxc2=1000;
        for(int j=0; j<CudaData->h_LxcSize[i]; j++){
            if(CudaData->h_Lxc[CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j] < 0)
                sgn = sgn*(-1);
            double temp = fabs(CudaData->h_Lxc[CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j]);
            if(temp < minLxc2){
                if(temp < minLxc1){
                    minLxc2 = minLxc1;
                    minLxc1 = temp;
                }
                else
                    minLxc2 = temp;
            }
        }
        for(int j=0; j<CudaData->h_LxcSize[i]; j++){
            if(fabs(CudaData->h_Lxc[CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j]) == minLxc1){
                if(CudaData->h_Lxc[CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j] < 0)
                    CudaData->h_Lcx[CudaData->h_LcxPosition[CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j]] = sgn*(-1) * minLxc2;
                else
                    CudaData->h_Lcx[CudaData->h_LcxPosition[CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j]] = sgn * minLxc2;
            }
            else{
                if(CudaData->h_Lxc[CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j] < 0)
                    CudaData->h_Lcx[CudaData->h_LcxPosition[CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j]] = sgn*(-1) * minLxc1;
                else
                    CudaData->h_Lcx[CudaData->h_LcxPosition[CudaData->h_LxcBegin[i]+CudaData->h_LxcStripe[i]*j]] = sgn * minLxc1;
            }
        }
    }
/*
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
    }*/
}

void CIRCUIT::UpdateLxc_CPU(){
    for(int i=0; i<n_Total; i++){
        double sumLcx=0;
        for(int j=0; j<CudaData->h_LcxSize[i]; j++){
            sumLcx = sumLcx + CudaData->h_Lcx[CudaData->h_LcxBegin[i]+CudaData->h_LcxStripe[i]*j];
        }
        for(int j=0; j<CudaData->h_LcxSize[i]; j++){
             CudaData->h_Lxc[CudaData->h_LxcPosition[CudaData->h_LcxBegin[i]+CudaData->h_LcxStripe[i]*j]] = CudaData->h_Lint[i] + (sumLcx - CudaData->h_Lcx[CudaData->h_LcxBegin[i]+CudaData->h_LcxStripe[i]*j]);
        }
    }
}

void CIRCUIT::Calculate_Posterior_CPU(){
    for(int i=0; i<n_Total; i++){
        double sumLcx=0;
        for(int j=0; j<CudaData->h_LcxSize[i]; j++)
            sumLcx = sumLcx + CudaData->h_Lcx[CudaData->h_LcxBegin[i]+CudaData->h_LcxStripe[i]*j];
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
    cudaMemcpy(CudaData->d_LcxStripe, CudaData->h_LcxStripe, n_Total * sizeof(int), cudaMemcpyHostToDevice);
    ////// CheckNode on device //////
    cudaMemcpy(CudaData->d_LxcSize, CudaData->h_LxcSize, k_Total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_LxcBegin, CudaData->h_LxcBegin, k_Total * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_LcxPosition, CudaData->h_LcxPosition, TotalEdge * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(CudaData->d_LxcStripe, CudaData->h_LxcStripe, k_Total * sizeof(int), cudaMemcpyHostToDevice);

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
__global__ void UpdateLcx_GPU(int* d_LxcSize, double* d_Lxc, int* d_LxcBegin, int* d_LcxPosition, double* d_Lcx,int k_Total, int* d_LxcStripe){
#else
__global__ void UpdateLcx_GPU(int* d_LxcSize, float* d_Lxc, int* d_LxcBegin, int* d_LcxPosition, float* d_Lcx,int k_Total, int* d_LxcStripe){
#endif
    int total_task = gridDim.x * blockDim.x;
    int task_sn = blockIdx.x*blockDim.x + threadIdx.x;
    //__shared__ double buffer[128];
    //double* sgnA = &buffer[0];
    //double* minLxcA = &buffer[64];
    
    for(int i=task_sn; i<k_Total; i+=total_task){
#ifdef DOUBLE
        double sgn=1;
        double minLxc1=1000;
        double minLxc2=1000;
#else
        float sgn=1;
        float minLxc1=1000;
        float minLxc2=1000;
#endif
        for(int j=0; j<d_LxcSize[i]; j++){
            if(d_Lxc[d_LxcBegin[i]+d_LxcStripe[i]*j] < 0)
                sgn = sgn*(-1);
#ifdef DOUBLE
            double temp = fabs(d_Lxc[d_LxcBegin[i]+d_LxcStripe[i]*j]);
#else
            float temp = fabs(d_Lxc[d_LxcBegin[i]+d_LxcStripe[i]*j]);
#endif
            if(temp < minLxc2){
                if(temp < minLxc1){
                    minLxc2 = minLxc1;
                    minLxc1 = temp;
                }
                else{
                    minLxc2 = temp;
                }
            }
        }
        for(int j=0; j<d_LxcSize[i]; j++){
            if(fabs(d_Lxc[d_LxcBegin[i]+d_LxcStripe[i]*j]) == minLxc1){
                if(d_Lxc[d_LxcBegin[i]+d_LxcStripe[i]*j] < 0)
                    d_Lcx[d_LcxPosition[d_LxcBegin[i]+d_LxcStripe[i]*j]] = sgn*(-1) * minLxc2;
                else
                    d_Lcx[d_LcxPosition[d_LxcBegin[i]+d_LxcStripe[i]*j]] = sgn * minLxc2;
            }
            else{
                if(d_Lxc[d_LxcBegin[i]+d_LxcStripe[i]*j] < 0)
                    d_Lcx[d_LcxPosition[d_LxcBegin[i]+d_LxcStripe[i]*j]] = sgn*(-1) * minLxc1;
                else
                    d_Lcx[d_LcxPosition[d_LxcBegin[i]+d_LxcStripe[i]*j]] = sgn * minLxc1;
            }
        }
    }

    /*for(int i=task_sn; i<k_Total; i+=total_task){
        for(int j=0; j<d_LxcSize[i]; j++){
            double sgn=1;
            double minLxc=1000;
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
    }*/
}
#ifdef DOUBLE
__global__ void UpdateLxc_GPU(int* d_LcxSize, double* d_Lcx, int* d_LcxBegin, int* d_LxcPosition, double* d_Lxc, double* d_Lint, int n_Total, int* d_LcxStripe){
#else
__global__ void UpdateLxc_GPU(int* d_LcxSize, float* d_Lcx, int* d_LcxBegin, int* d_LxcPosition, float* d_Lxc, float* d_Lint, int n_Total, int* d_LcxStripe){
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
            sumLcx = sumLcx + d_Lcx[d_LcxBegin[i]+d_LcxStripe[i]*j];
        }
        for(int j=0; j<d_LcxSize[i]; j++){
             d_Lxc[d_LxcPosition[d_LcxBegin[i]+d_LcxStripe[i]*j]] = d_Lint[i] + (sumLcx - d_Lcx[d_LcxBegin[i]+d_LcxStripe[i]*j]);
        }
    }
}
#ifdef DOUBLE
__global__ void Calculate_Posterior_GPU(int* d_LcxSize, double* d_Lcx, int* d_LcxBegin, double* d_Lint, int* d_DecodedData,int n_Total, int* d_LcxStripe){
#else
__global__ void Calculate_Posterior_GPU(int* d_LcxSize, float* d_Lcx, int* d_LcxBegin, float* d_Lint, int* d_DecodedData,int n_Total, int* d_LcxStripe){
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
            sumLcx = sumLcx + d_Lcx[d_LcxBegin[i]+d_LcxStripe[i]*j];
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
