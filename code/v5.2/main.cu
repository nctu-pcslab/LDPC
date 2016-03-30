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
    //cout<<"test9"<<endl;
    Circuit.CUDA_MemoryAllocate();
    //cout<<"test10"<<endl;
    Circuit.CUDA_CreateDataArray();
    //cout<<"test11"<<endl;
    Circuit.MemoryCopy_H2D();
    //cout<<"test12"<<endl;
    /////////////////////////////
    //for(int i=0; i<SNR_Size; i++){
    for(int i=0 ; i< atoi(argv[9]); i++){
        Circuit.SNR_Index=i;            //Select SNR
        //cout<<"SNR = "<<Circuit.SNR[Circuit.SNR_Index]<<endl;
        Circuit.ResetValue();
        for(int j=0; j<Circuit.PatternNum; j++){
            //cout<<"test5"<<endl;
            Circuit.GenerateMessage();
            //cout<<"test6"<<endl;
            Circuit.EncodeMessage();
            //cout<<"test7"<<endl;
            Circuit.AWGN_Channel();
            //cout<<"test8"<<endl;
            Circuit.Reset_Lxc_Lcx();
            
            Circuit.Timer.Begin();
            ///////// CUDA ////////
            //cout<<"test"<<endl;
            Circuit.CUDA_CreateDataArray2();
            //cout<<"test2"<<endl;
            Circuit.MemoryCopy_H2D2();
            ///////////////////////
            //cout<<"test3"<<endl;
            Circuit.DecodeMessage();
            ExcutionTime = ExcutionTime + Circuit.Timer.End(); 
            //cout<<"test4"<<endl; 
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
