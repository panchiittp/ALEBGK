#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <chrono>
#include <limits>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <set>
#include <iomanip>
#include "Definitions/Definitions2D.hpp"
#include "Parameters/Parameters.hpp"
#include "SaveParticles/SaveParticles.hpp"
#include "ParticleManagement/ParticleManagement.hpp"
#include "MemoryManagement/MemoryManagement.hpp"
#include "InitialConditions/InitialConditions.hpp"
#include "Interpolation/Interpolation.hpp"

using namespace std;


int main()
{
    auto startprog = std::chrono::high_resolution_clock::now();
    readparameters();
    CalculateParameters();
    showparameters();

    
    bool meminfo = true;


    if (meminfo)
        getfreememinfo("Initial");
    int blockSize = 512;
    BGKParticle *dP;
    voxelDetails *dvoxinfo;// *ddelpart;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMallocManaged((void **)&dP, sizeof(BGKParticle) * CalcParam.N);
    cudaMemPrefetchAsync(dP, CalcParam.N* sizeof(BGKParticle), cudaCpuDeviceId);

    cudaMallocManaged((void **)&dvoxinfo, sizeof(voxelDetails) * CalcParam.nbxBox * CalcParam.nbyBox * CalcParam.nbzBox);
    //cudaMallocManaged((void **)&ddelpart, sizeof(voxelDetails) * 5);
    cudaMemPrefetchAsync(dvoxinfo, CalcParam.nbxBox * CalcParam.nbyBox * CalcParam.nbzBox* sizeof(voxelDetails), cudaCpuDeviceId);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for Memory Allocations : " << milliseconds/1000.0f << " seconds==========" << std::endl;

    int numBlocks = (CalcParam.N + blockSize - 1) / blockSize;
    if (meminfo)
        getfreememinfo("Initial  Allocation");
    cudaEventRecord(start, 0);        
    GenerateParticlesKernel<<<numBlocks, blockSize>>>(dP, CalcParam, Param, Domain);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "=====Time for  Particle Generation: " << milliseconds/1000.0f << " seconds======" << std::endl;

    std::cout << "Particle Generation completed succesfully" << std::endl;
    if (meminfo)
        getfreememinfo("Particle Generation");

    cudaEventRecord(start, 0);        
    cudaDeviceSynchronize();

        
    updateVoxelNumberingKernel<<<numBlocks, blockSize>>>(dP, CalcParam, Domain, dvoxinfo);
    std::cout << "Voxel formation completed succesfully" << std::endl;
    // iwantsaveonly(dP, CalcParam,0, dvoxinfo, true, "NeighboursMatMul.txt");
    //     return 0;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Time for Memory Voxel Number: " << milliseconds/1000.0f << " seconds==========" << std::endl;

    printsizerequired(sizeof(BGKParticle) * CalcParam.N);
    if (meminfo)
        getfreememinfo("Before Update Neighbours");

    cudaEventRecord(start, 0);        

    // updateNeighboursKernel<<<numBlocks, blockSize>>>(dP, CalcParam);
    findNeighborParticlesPeriodic<<<numBlocks, blockSize>>>(dP, CalcParam, dvoxinfo,Domain);
    cout << "updating neighbours are completed succesfully" << endl;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    // IdentifyNeighbourType<<<numBlocks, blockSize>>>(dP, CalcParam,Domain);
    // cudaDeviceSynchronize();
    printneighanddist(0,dP);
    printneighvoxel(0,dP);    
    SaveNeighbourParticleForMatlab("NeighbourInitialParticles.dat",dP,CalcParam.N,221);

    cout << "updating neighbours are completed succesfully" << endl;

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    IdentifyNeighbourType<<<numBlocks, blockSize>>>(dP, CalcParam,Domain);
    cudaDeviceSynchronize();
    //iwantprintsave(dP, CalcParam,0, dvoxinfo, true,false);
    //return;
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for Neighbour Search: " << milliseconds/1000.0f << " seconds==========" << std::endl;

    if (meminfo)
        getfreememinfo("Update Neighbours");

    cudaEventRecord(start, 0);        

    applyInitialConditionsKernel<<<numBlocks, blockSize>>>(dP, CalcParam, Param, IC, Constant);
    
    cudaDeviceSynchronize();
    std::cout<<"Synchronizing the Device"<<std::endl;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time for Applying Initial Conditions: " << milliseconds/1000.0f << " seconds==========" << std::endl;
    cudaDeviceSynchronize();
    std::string direction[]={"Left","Right","Bottom","Top","Front","Back"};
    for(int i=0;i<4;i++)
    {
        SavePeriodicNeighbourParticleForMatlab(direction[i]+"PeriodicNeighbourInitialParticles.dat",dP,CalcParam.N,221,i);
    }
    
    
    if (meminfo)
        getfreememinfo("Initial Coniditons");
    cout << "Applying Initial Conditions are completed succesfully" << endl;       
    SaveParticleForMatlab("InitialParticles.dat",dP,CalcParam.N);
    int t=0;
    int count=0;
    while (t < Param.tfinal)
    {
        std::cout << "Working on Time Step : " << t << " and Iteration Number: " << count << std::endl;
        auto start1 = std::chrono::high_resolution_clock::now();

        const int blockSize1 = 512;
        int numBlocks1 = (CalcParam.N + blockSize1 - 1) / blockSize1;
        cudaEventRecord(start, 0);        
        
        std::cout << "Working on MLS Method Kernel" << std::endl;
        
        for(int flag=0;flag<5;flag++)
        {    
//            int flag=1;
            ConstructCenterMMatrixKernel<<<numBlocks1, blockSize1>>>(dP, Param, CalcParam, Constant,Domain,flag);
            cudaDeviceSynchronize();
        }
 
        

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Completed MLS Method  Kernel" << std::endl;
        std::cout << "Time for MLS Method  Kernel: " << milliseconds/1000.0f << " seconds==========" << std::endl;
        std::cout<<"Param.Nv = "<<Param.Nv<<std::endl;
        for(int i=0;i<CalcParam.N;i++)
            if(dP[i].boundary!=true)
                printperiodicneigh(dP,CalcParam,i,"PeriodicNeighbours.txt"); 
        return 0;
    }
}