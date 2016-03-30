#include <iostream>
#include <math.h>
#include "circuit.h"
using namespace std;
CIRCUIT Circuit;

int main(int argc, char ** argv){
    Circuit.Parser(argv);
    float ExcutionTime=0;
    //srand(time(NULL));

    ////////// CUDA ////////////
    Circuit.CUDA_MemoryAllocate();
    Circuit.CUDA_CreateDataArray();
    Circuit.MemoryCopy_H2D();
    /////////////////////////////
    //for(int i=0; i<SNR_Size; i++){
    for(int i=0; i< atoi(argv[9]); i++){
        Circuit.SNR_Index=i;            //Select SNR
        Circuit.ResetValue();
        for(int j=0; j<Circuit.PatternNum; j++){
            Circuit.GenerateMessage();
            Circuit.EncodeMessage();
            Circuit.AWGN_Channel();
            Circuit.Reset_Lxc_Lcx();
            
            Circuit.Timer.Begin();
            ///////// CUDA ////////
            Circuit.CUDA_CreateDataArray2();
            Circuit.MemoryCopy_H2D2();
            ///////////////////////
            Circuit.DecodeMessage();
            ExcutionTime = ExcutionTime + Circuit.Timer.End(); 
            
            Circuit.Calculate_Error();
        }
        //cout<<"BER = "<<Circuit.ErrorNum/(2978*m)<<"    ErrorNum="<<Circuit.ErrorNum<<endl;
        cout<<Circuit.ErrorNum/(Circuit.PatternNum*Circuit.m_Total)<<endl;
    }
    //cout<<Circuit.Timer.End()<<" seconds"<<endl;
    cout<<"Total Decoding Time = "<<ExcutionTime/CLOCKS_PER_SEC<<" seconds"<<endl;
    //cout<<"Throughput = "<<(Circuit.PatternNum*Circuit.m_Total*SNR_Size)/1000000/(ExcutionTime/CLOCKS_PER_SEC)<<" Mbps"<<endl;
    cout<<"Throughput = "<<(Circuit.PatternNum*Circuit.m_Total*atoi(argv[9]))/1000000/(ExcutionTime/CLOCKS_PER_SEC)<<" Mbps"<<endl;
#ifdef PROFILE
    Circuit.Timer.RunTimeProfile(); 
#endif
    return 0;
}
