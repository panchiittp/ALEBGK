#ifndef PARTICLEMANAGEMENT_HPP
#define PARTICLEMANAGEMENT_HPP

#include "../Definitions/Definitions2D.hpp"

__global__ void GenerateParticlesKernel(BGKParticle *dP, CalcParameters CalcParam, Parameters Param, DomainBoundary Domain)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // while (i < CalcParam.N)
    // while(i<100)
    if (i < CalcParam.N)
    {
        int y = i % Param.Nx;
        int x = i / Param.Nx;

        dP[i].pos.x = Domain.xleft + (double)x * CalcParam.dx;
        dP[i].pos.y = Domain.ybottom + (double)y * CalcParam.dy;

        if (dP[i].pos.x == Domain.xleft || dP[i].pos.x == Domain.xright ||
            dP[i].pos.y == Domain.ybottom || dP[i].pos.y == Domain.ytop)
            dP[i].boundary = true;

        dP[i].totneigh = 0;
        dP[i].active = true;

        for (int k = 0; k < 3 * 50; ++k)
        {
            dP[i].neightype[k] = -1;            
        }     
    }
}

void GenerateParticlesCPU(BGKParticle *dP, CalcParameters CalcParam, Parameters Param, DomainBoundary Domain)
{
    

    // while (i < CalcParam.N)
    // while(i<100)
    for (int i=0;i < CalcParam.N;i++)
    {
        int y = i % Param.Nx;
        int x = i / Param.Nx;

        dP[i].pos.x = Domain.xleft + (double)x * CalcParam.dx;
        dP[i].pos.y = Domain.ybottom + (double)y * CalcParam.dy;

        if (dP[i].pos.x == Domain.xleft || dP[i].pos.x == Domain.xright ||
            dP[i].pos.y == Domain.ybottom || dP[i].pos.y == Domain.ytop)
            dP[i].boundary = true;

        dP[i].totneigh = 0;
        dP[i].active = true;

        for (int k = 0; k < 3 * 50; ++k)
        {
            dP[i].neightype[k] = -1;            
        }        
    }
}

#include "VoxelManagement.hpp"
#include "NeighbourSearch.hpp"

#endif
