#ifndef CPUSOLVERS_HPP
#define CPUSOLVERS_HPP

void CPUSolvers()
{

    BGKParticle *dP;
    voxelDetails *dvoxinfo;// *ddelpart;
    dP=new BGKParticle[CalcParam.N];
    dvoxinfo=new voxelDetails[CalcParam.nbxBox * CalcParam.nbyBox * CalcParam.nbzBox];
    GenerateParticlesCPU(dP, CalcParam, Param, Domain);
    GenerateVoxelNumberingCPU(dP, CalcParam, Domain, dvoxinfo);
    updateVoxelNumberingCPU(dP, CalcParam, Domain, dvoxinfo);
    printallvoxel(dvoxinfo,CalcParam);
    findNeighborParticlesPeriodicCPU(dP, CalcParam, dvoxinfo,Domain);
    std::cout << "updating neighbours are completed succesfully" << std::endl;
    printallparticleneigh(dP,CalcParam,"AllParticleInformation.dat");
    IdentifyNeighbourTypeCPU(dP, CalcParam,Domain);
    printneighanddist(visParticleNumber,dP);
    printneighvoxel(visParticleNumber,dP);    
    SaveNeighbourParticleForMatlab("NeighbourInitialParticlesCPU.dat",dP,CalcParam.N,visParticleNumber);

    std::cout << "updating neighbours Type are completed succesfully" << std::endl;

    
    
    applyInitialConditionsCPU(dP, CalcParam, Param, IC, Constant);



    std::string direction[]={"Left","Right","Bottom","Top","Front","Back"};
    for(int i=0;i<4;i++)
    {
        SavePeriodicNeighbourParticleForMatlab(direction[i]+"PeriodicNeighbourInitialParticlesCPU.dat",dP,CalcParam.N,visParticleNumber,i);
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

    
        std::cout << "Working on MLS Method " << std::endl;
        
        // for(int flag=0;flag<5;flag++)
        // {    
            int flag=4;
            ConstructCenterMMatrixCPU(dP, Param, CalcParam, Constant,Domain,flag);
        // }

        

         printperiodicneigh(dP,CalcParam,visParticleNumber,"PeriodicNeighboursCPU.txt"); 
    //     // for(int i=0;i<CalcParam.N;i++)
    //     //     if(dP[i].boundary!=true)
    //     //         printperiodicneigh(dP,CalcParam,i,"PeriodicNeighbours.txt"); 
         return;
    }
}
#endif