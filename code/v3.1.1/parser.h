#ifndef PARSER_H
#define PARSER_H

class TIMER{
    public:
        float Begin_Clock;
        void Begin();
        float End();
        /////// detail profile //////
        TIMER():tCUDA_MemoryAllocate(0),tCUDA_CreateDataArray(0),tCUDA_CreateDataArray2(0),tMemoryCopy_H2D(0),tMemoryCopy_H2D2(0),tMemoryCopy_D2H(0),tUpdateLcx(0),tUpdateLxc(0),tCalculate_Posterior(0),tCheck_Syndrome(0){}
        float tCUDA_MemoryAllocate;
        float tCUDA_CreateDataArray;
        float tCUDA_CreateDataArray2;
        float tMemoryCopy_H2D;
        float tMemoryCopy_H2D2;
        float tMemoryCopy_D2H;
        float tUpdateLcx;
        float tUpdateLxc;
        float tCalculate_Posterior;
        float tCheck_Syndrome;
        cudaEvent_t start;
        cudaEvent_t stop;
        void TimerStart();
        void TimerFinish(float& tTime);
        void RunTimeProfile();
};
#endif
