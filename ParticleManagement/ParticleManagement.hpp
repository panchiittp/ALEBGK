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

        dP[i].x = Domain.xleft + (double)x * CalcParam.dx;
        dP[i].y = Domain.ybottom + (double)y * CalcParam.dy;

        if (dP[i].x == Domain.xleft || dP[i].x == Domain.xright ||
            dP[i].y == Domain.ybottom || dP[i].y == Domain.ytop)
            dP[i].boundary = true;

        dP[i].totneigh = 0;
        dP[i].active = true;

        for (int k = 0; k < 3 * 180; ++k)
        {
            dP[i].neightype[k] = -1;            
        }        
    }
}
#include "VoxelManagement.hpp"
#include "NeighbourSearch.hpp"

#endif
