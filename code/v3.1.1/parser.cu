#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include "circuit.h"
#include "parser.h"
using namespace std;

void CIRCUIT::Parser(char** argv){
    cout<<"Parsing Matrix"<<endl;
    m=atoi(argv[3]);
    n=atoi(argv[4]);
    k=atoi(argv[5]);
    ITERATION=atoi(argv[6]);
    PatternNum=atoi(argv[7]);
    NumOfCodeWord=atoi(argv[8]);
    m_Total = m * NumOfCodeWord;
    n_Total = n * NumOfCodeWord;
    k_Total = k * NumOfCodeWord;
    ///////// initialize data structure ///////
    MessageData = new int[m_Total];
    DecodedData = new int[n_Total];
#ifdef DOUBLE
    EncodedData = new double[n_Total];
    ReceivedData = new double[n_Total];
#else
    EncodedData = new float[n_Total];
    ReceivedData = new float[n_Total];
#endif

    for(int i=0;i<m_Total;i++){
        CHECKNODE* pNewCheckNode(0);
        pNewCheckNode = new CHECKNODE;
        pNewCheckNode->ID=i;
        CheckNode.push_back(pNewCheckNode);
    }
    int SIZE;
    if(n_Total >= k_Total)  SIZE = n_Total;
    else                    SIZE = k_Total;
    for(int i=0;i<SIZE;i++){
        BITNODE* pNewBitNode(0);
        pNewBitNode = new BITNODE;
        pNewBitNode->ID=i;
        BitNode.push_back(pNewBitNode);
    }
    //////// parsing G matrix //////
    string str;
    ifstream opfile;
    opfile.open(argv[1]);
    if(!opfile){
        cout<<"Open Matrix_G Fail! Abort Program!"<<endl;
        assert(0);
    }
    for(int j=0; j<m; j++){
        for(int i=0; i<n; i++){
            opfile>>str;
            if(str == "1"){
                for(int ii=0; ii<NumOfCodeWord; ii++){
                    CheckNode[j+ii*m]->BitNode_G.push_back(BitNode[i+ii*n]);
                    BitNode[i+ii*n]->CheckNode_G.push_back(CheckNode[j+ii*m]);
                }
            }
        }
    }
    /*cout<<CheckNode[0]->BitNode_G.size()<<endl;
    for(unsigned i=0;i<BitNode.size();i++)
        cout<<BitNode[i]->CheckNode_G.size()<<" ";
    cout<<endl;*/
    //for(unsigned i=0;i<CheckNode.size();i++)
    //    cout<<i<<" "<<CheckNode[i]->BitNode_G.size()<<endl;
    opfile.close();
    //////// parsing H matrix //////
    opfile.open(argv[2]);
    if(!opfile){
        cout<<"Open Matrix_H Fail! Abort Program!"<<endl;
        assert(0);
    }
    /*for(unsigned i=0; i<CheckNode.size(); i++)
        cout<<CheckNode[i]->BitNode_H.size()<<" ";
    cout<<endl;*/
    /*for(unsigned i=0; i<BitNode.size(); i++)
        cout<<BitNode[i]->CheckNode_H.size()<<" ";
    cout<<endl;*/
    for(int j=0; j<k; j++){
        for(int i=0; i<n; i++){
            opfile>>str;
            //cout<<str<<" ";
            if(str == "1"){
                for(int ii=0; ii<NumOfCodeWord; ii++){
                    CheckNode[j+ii*k]->Lcx_position.push_back(BitNode[i+ii*n]->Lcx.size());
                    BitNode[i+ii*n]->Lxc_position.push_back(CheckNode[j+ii*k]->Lxc.size());

                    CheckNode[j+ii*k]->BitNode_H.push_back(BitNode[i+ii*n]);
                    CheckNode[j+ii*k]->Lxc.push_back(0);
                    BitNode[i+ii*n]->CheckNode_H.push_back(CheckNode[j+ii*k]);
                    BitNode[i+ii*n]->Lcx.push_back(0);
                    //BitNode[i]->Lint.push_back(0);
                }
            }
        }
        //cout<<endl;
    }
    /*for(unsigned i=0; i<CheckNode.size(); i++)
        cout<<CheckNode[i]->BitNode_H.size()<<" ";
    cout<<endl;*/
    /*for(unsigned i=0; i<BitNode.size(); i++)
        cout<<BitNode[i]->CheckNode_H.size()<<" ";
    cout<<endl;*/
    opfile.close();

    TotalEdge=0;
    for(unsigned i=0; i<n_Total; i++)
        TotalEdge = TotalEdge + BitNode[i]->CheckNode_H.size();
    /////// parser finished //////
    cout<<"Size of G_matrix = "<<m<<" * "<<n<<endl;
    cout<<"Size of H_matrix = "<<k<<" * "<<n<<endl;
    cout<<"# of CheckNode = "<<k_Total<<endl;
    cout<<"# of BitNode = "<<BitNode.size()<<endl;
    cout<<"# of Edge = "<<TotalEdge<<endl;
    cout<<"ITERATION = "<<ITERATION<<endl;
    cout<<"PatternNum = "<<PatternNum<<endl;
    cout<<"NumOfCodeWord = "<<NumOfCodeWord<<endl;
#ifdef DOUBLE
    cout<<"\e[32mDOUBLE precision\e[m"<<endl;
#else
    cout<<"\e[32mFLOAT precision\e[m"<<endl;
#endif
}

void TIMER::Begin(){
    Begin_Clock = clock();
}

float TIMER::End(){
    //return ( (double)( clock() - Begin_Clock ) / (double)CLOCKS_PER_SEC );
    return (clock() - Begin_Clock);
}

void TIMER::TimerStart(){
    cudaEventCreate (&start);
    cudaEventCreate (&stop);
    cudaEventRecord(start, 0);
}

void TIMER::TimerFinish(float& tTime){
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    tTime = tTime + elapsedTime;
}

void TIMER::RunTimeProfile(){
    /*cout<<"CUDA_MemoryAllocate time:\t"<<tCUDA_MemoryAllocate<<" ms"<<endl;
    cout<<"CUDA_CreateDataArray time:\t"<<tCUDA_CreateDataArray<<" ms"<<endl;
    cout<<"\e[32mCUDA_CreateDataArray2 time:\t"<<tCUDA_CreateDataArray2<<" ms\e[m"<<endl;
    cout<<"MemoryCopy_H2D time:\t\t"<<tMemoryCopy_H2D<<" ms"<<endl;
    cout<<"\e[32mMemoryCopy_H2D2 time:\t\t"<<tMemoryCopy_H2D2<<" ms"<<endl;
    cout<<"MemoryCopy_D2H time:\t\t"<<tMemoryCopy_D2H<<" ms"<<endl;
    cout<<"UpdateLcx time:\t\t\t"<<tUpdateLcx<<" ms"<<endl;
    cout<<"UpdateLxc time:\t\t\t"<<tUpdateLxc<<" ms"<<endl;
    cout<<"Calculate_Posterior time:\t"<<tCalculate_Posterior<<" ms"<<endl;
    cout<<"Check_Syndrome:\t\t\t"<<tCheck_Syndrome<<" ms\e[m"<<endl;*/
    cout<<tCUDA_CreateDataArray2<<endl;
    cout<<tMemoryCopy_H2D2<<endl;
    cout<<tMemoryCopy_D2H<<endl;
    cout<<tUpdateLcx<<endl;
    cout<<tUpdateLxc<<endl;
    cout<<tCalculate_Posterior<<endl;
    cout<<tCheck_Syndrome<<endl;
}
