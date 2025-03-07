#ifndef CPUSOLVERS_HPP
#define CPUSOLVERS_HPP

void CPUSolvers()
{

    BGKParticle *dP;
    voxelDetails *dvoxinfo;// *ddelpart;
    dP=new BGKParticle[CalcParam.N];
    dvoxinfo=new voxelDetails[CalcParam.nbxBox * CalcParam.nbyBox * CalcParam.nbzBox];
    GenerateParticlesKernelCPU(dP, CalcParam, Param, Domain);
    
    updateVoxelNumberingKernelCPU(dP, CalcParam, Domain, dvoxinfo);
    findNeighborParticlesPeriodicCPU(dP, CalcParam, dvoxinfo,Domain);
    cout << "updating neighbours are completed succesfully" << endl;
    IdentifyNeighbourTypeCPU(dP, CalcParam,Domain);
    printneighanddist(0,dP);
    printneighvoxel(0,dP);    
    SaveNeighbourParticleForMatlab("NeighbourInitialParticlesCPU.dat",dP,CalcParam.N,221);

    cout << "updating neighbours are completed succesfully" << endl;

    
    
    applyInitialConditionsKernelCPU(dP, CalcParam, Param, IC, Constant);



    std::string direction[]={"Left","Right","Bottom","Top","Front","Back"};
    for(int i=0;i<4;i++)
    {
        SavePeriodicNeighbourParticleForMatlab(direction[i]+"PeriodicNeighbourInitialParticlesCPU.dat",dP,CalcParam.N,221,i);
    }
    
    
    // if (meminfo)
    //     getfreememinfo("Initial Coniditons");
    cout << "Applying Initial Conditions are completed succesfully" << endl;       
    SaveParticleForMatlab("InitialParticlesCPU.dat",dP,CalcParam.N);
    int t=0;
    int count=0;
    while (t < Param.tfinal)
    {
        std::cout << "Working on Time Step : " << t << " and Iteration Number: " << count << std::endl;
        auto start1 = std::chrono::high_resolution_clock::now();

    
        std::cout << "Working on MLS Method Kernel" << std::endl;
        
        // for(int flag=0;flag<5;flag++)
        // {    
            int flag=4;
            ConstructCenterMMatrixKernelCPU(dP, Param, CalcParam, Constant,Domain,flag);
        // }

        

         printperiodicneigh(dP,CalcParam,221,"PeriodicNeighboursCPU.txt"); 
    //     // for(int i=0;i<CalcParam.N;i++)
    //     //     if(dP[i].boundary!=true)
    //     //         printperiodicneigh(dP,CalcParam,i,"PeriodicNeighbours.txt"); 
         return;
    }
}
#endif