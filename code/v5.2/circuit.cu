#include <iostream>
#include <math.h>
#include <algorithm>
#include "circuit.h"
using namespace std;

void CIRCUIT::ResetValue(){
    ErrorNum = 0;
    Syndrome = 1;
#ifdef DOUBLE
    double power=pow(10,(SNR[SNR_Index]/10));
#else
    float power=pow(10,(SNR[SNR_Index]/10));
#endif
    Variance=sqrt( (n/(2*m)) / power );
    //cout<<"SNR = "<<SNR[SNR_Index]<<endl;
}

void CIRCUIT::GenerateMessage(){
    //srand(time(0));
    for(int i=0; i<m_Total; i++)
        MessageData[i]=(int)(2.0*rand()/(RAND_MAX+1.0));
    /*for(int i=0; i<m; i++)
        cout<<MessageData[i]<<" ";
    cout<<endl;*/
}

void CIRCUIT::EncodeMessage(){
    for(unsigned i=0; i<BitNode.size(); i++){
        int temp=0;
        for(unsigned j=0; j<BitNode[i]->CheckNode_G.size(); j++){
            //cout<<BitNode[i]->CheckNode_G[j]->ID<<" ";
            temp = (temp + MessageData[BitNode[i]->CheckNode_G[j]->ID])%2;
        }
        if(temp==0)
            EncodedData[i]=1;
        else if(temp==1)
            EncodedData[i]=-1;
        //cout<<temp<<" "<<EncodedData[i]<<endl;;
    }
}

void CIRCUIT::AWGN_Channel(){
    //double power=pow(10,(SNR[index]/10));
    //Variance=sqrt( (n/(2*m)) / power );
    //cout<<power<<" "<<variance<<endl;
    //cout<<"SNR = "<<SNR[index]<<endl;
    for(int i=0; i<n_Total; i++){
        Channel_GaussianGen();
        //cout<<GaussNum.x1<<" "<<GaussNum.x2<<endl;
        ReceivedData[i]=EncodedData[i]+Variance*GaussNum.x1;
        //cout<<ReceivedData[i]<<" ";
    }
    //cout<<endl;
}

void CIRCUIT::Reset_Lxc_Lcx(){
    for(unsigned i=0; i<BitNode.size(); i++){
        BitNode[i]->Lint = 2*ReceivedData[i]/(Variance*Variance);
        //for(unsigned j=0; j<BitNode[i]->Lcx.size(); j++){
            //BitNode[i]->Lcx[j] = 0;
            //BitNode[i]->Lint[j] = 2*ReceivedData[i]/(Variance*Variance);
            //BitNode[i]->Lxc[j] = BitNode[i]->Lint[j];
        //}
        /*if(BitNode[i]->CheckNode_H.size() != BitNode[i]->Lcx.size())
            cout<<"Something wrong"<<endl;
        if(BitNode[i]->Lcx.size() != BitNode[i]->Lint.size())
            cout<<"Something wrong"<<endl;
        if(BitNode[i]->Lcx.size() != BitNode[i]->Lxc_position.size())
            cout<<"Something wrong"<<endl;*/
    }
    for(unsigned i=0; i<k_Total; i++){
        for(unsigned j=0; j<CheckNode[i]->Lxc.size(); j++){
            //int index = CheckNode[i]->Lcx_position[j];
            CheckNode[i]->Lxc[j] = CheckNode[i]->BitNode_H[j]->Lint;
        }
        /*if(CheckNode[i]->BitNode_H.size() != CheckNode[i]->Lxc.size())
            cout<<"Something wrong"<<endl;
        if(CheckNode[i]->Lxc.size() != CheckNode[i]->Lcx_position.size())
            cout<<"Something wrong"<<endl;*/
    }
}

void CIRCUIT::DecodeMessage(){
    int iter=0;
    //cout<<iter<<" "<<Syndrome<<endl;
    Syndrome=1;
    while(Syndrome!=0 && iter<ITERATION){
#ifdef PROFILE
    Timer.TimerStart();
#endif
        //UpdateLcx();
        //UpdateLcx_CPU();
        UpdateLcx_GPU<<<UCX_BLK,UCX_THD>>>(CudaData->d_LxcSize, CudaData->d_Lxc, CudaData->d_LxcBegin, CudaData->d_LcxPosition, CudaData->d_Lcx, k_Total, CudaData->d_LxcStripe);
#ifdef PROFILE
    Timer.TimerFinish(Timer.tUpdateLcx);
#endif
#ifdef PROFILE
    Timer.TimerStart();
#endif  
        //UpdateLxc();
        //UpdateLxc_CPU();
        UpdateLxc_GPU<<<UXC_BLK,UXC_THD>>>(CudaData->d_LcxSize, CudaData->d_Lcx, CudaData->d_LcxBegin, CudaData->d_LxcPosition, CudaData->d_Lxc, CudaData->d_Lint, n_Total, CudaData->d_LcxStripe);
#ifdef PROFILE
    Timer.TimerFinish(Timer.tUpdateLxc);
#endif

        //cout<<iter<<endl;
        iter++;
    }
#ifdef PROFILE
    Timer.TimerStart();
#endif  
        //Calculate_Posterior();
        //Calculate_Posterior_CPU();
        Calculate_Posterior_GPU<<<UXC_BLK,UXC_THD>>>(CudaData->d_LcxSize, CudaData->d_Lcx, CudaData->d_LcxBegin, CudaData->d_Lint, CudaData->d_DecodedData, n_Total, CudaData->d_LcxStripe);
#ifdef PROFILE
    Timer.TimerFinish(Timer.tCalculate_Posterior);
#endif  
        /*for(int i=0; i<n_Total; i++)
            DecodedData[i] = CudaData->h_DecodedData[i];*/
        MemoryCopy_D2H();
        //Debug();
#ifdef PROFILE
    Timer.TimerStart();
#endif        
        Check_Syndrome();
#ifdef PROFILE
    Timer.TimerFinish(Timer.tCheck_Syndrome);
#endif
}

void CIRCUIT::UpdateLcx(){
    for(int i=0; i<k_Total; i++){
        for(unsigned j=0; j< CheckNode[i]->BitNode_H.size(); j++){
#ifdef DOUBLE
            double sgn=1;
            double minLxc=1000;
#else
            float sgn=1;
            float minLxc=1000;
#endif
            for(unsigned jj=0; jj< CheckNode[i]->BitNode_H.size(); jj++){
                if(j != jj){
                    if(CheckNode[i]->Lxc[jj] > 0)
                        sgn = sgn*1;
                    else
                        sgn = sgn*(-1);
                    minLxc = min(minLxc, fabs(CheckNode[i]->Lxc[jj]));
                }
            }
            int index = CheckNode[i]->Lcx_position[j];
            CheckNode[i]->BitNode_H[j]->Lcx[index] = sgn * minLxc;
        }
    }
}

void CIRCUIT::UpdateLxc(){
    for(int i=0; i<n_Total; i++){
#ifdef DOUBLE
        double sumLcx=0;
#else
        float sumLcx=0;
#endif
        for(unsigned j=0; j< BitNode[i]->CheckNode_H.size(); j++){
            sumLcx = sumLcx + BitNode[i]->Lcx[j];
        }
        for(unsigned j=0; j< BitNode[i]->CheckNode_H.size(); j++){
            int index = BitNode[i]->Lxc_position[j];
            BitNode[i]->CheckNode_H[j]->Lxc[index] = BitNode[i]->Lint + (sumLcx - BitNode[i]->Lcx[j]);
        }
    }
}

void CIRCUIT::Calculate_Posterior(){
    for(int i=0; i<n_Total; i++){
#ifdef DOUBLE
        double sumLcx=0;
#else
        float sumLcx=0;
#endif
        for(unsigned j=0; j< BitNode[i]->CheckNode_H.size(); j++)
            sumLcx = sumLcx + BitNode[i]->Lcx[j];
        if(BitNode[i]->Lint + sumLcx >= 0)
            DecodedData[i] = 0;
        else
            DecodedData[i] = 1;
    }
}

void CIRCUIT::Check_Syndrome(){
    Syndrome=0;
    for(int i=0; i<k_Total; i++){
        for(unsigned j=0; j<CheckNode[i]->BitNode_H.size(); j++){
            Syndrome = (Syndrome + DecodedData[CheckNode[i]->BitNode_H[j]->ID])%2;
        }
        //Syndrome=0;
        if(Syndrome!=0){
            //cout<<i<<endl;
            //i=m;
            break;
        }
        //cout<<i<<endl;
    }
}

void CIRCUIT::Calculate_Error(){
    /*for(int i=0; i<m; i++){
        if(DecodedData[i] != MessageData[i])
            ErrorNum = ErrorNum + 1;
    }*/
    for(int i=0; i<NumOfCodeWord; i++){
        for(int j=0; j<m ;j++){
            if(DecodedData[n*i+j] != MessageData[m*i+j])
                ErrorNum = ErrorNum + 1;
        }
    }
}
