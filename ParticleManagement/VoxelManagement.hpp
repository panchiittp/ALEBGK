
#ifndef VOXELMANAGEMENT_HPP
#define VOXELMANAGEMENT_HPP

///////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ void VoxelNumber(BGKParticle &dP, CalcParameters CalcParam, DomainBoundary Domain)
{
    double xBoxSize = CalcParam.xBox;
    double yBoxSize = CalcParam.yBox;

    int xBox = (int)floor((dP.x - Domain.xleft) / xBoxSize);   // column index of voxel
    int yBox = (int)floor((dP.y - Domain.ybottom) / yBoxSize); // row index of voxel

    if (dP.x == Domain.xright)
        xBox = CalcParam.nbxBox - 1;
    if (dP.y == Domain.ytop)
        yBox = CalcParam.nbyBox - 1;

    dP.voxel = xBox + CalcParam.nbxBox * yBox;
    // return hBox + calcParam.nbhBox * vBox;
}

__device__ void VoxelInformation(BGKParticle &dP, CalcParameters CalcParam)
{
    int voxIndex = dP.voxel;
    int x = voxIndex % CalcParam.nbxBox;
    int y = voxIndex / CalcParam.nbxBox;

    int count = 0;

    // Iterate over the possible neighbors in 2D
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            // Apply periodic boundary conditions
            int nx = (x + dx + CalcParam.nbxBox) % CalcParam.nbxBox;
            int ny = (y + dy + CalcParam.nbyBox) % CalcParam.nbyBox;

            // Calculate the 1D index of the neighbor voxel
            int neighborIndex = nx + ny * CalcParam.nbxBox;
            dP.neighVoxel[count++] = neighborIndex;
        }
    }

    dP.totvoxel = count;
}

//////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void updateVoxelNumberingKernel(BGKParticle *dP, CalcParameters CalcParam, DomainBoundary Domain, voxelDetails *voxinfo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < CalcParam.N)
    {
        VoxelNumber(dP[i], CalcParam, Domain);
        int voxindex = dP[i].voxel;
        int count = atomicAdd(&voxinfo[voxindex].count, 1);
        voxinfo[voxindex].particleindex[count] = i;
        VoxelInformation(dP[i], CalcParam);
        __syncthreads();
        // i += blockDim.x * gridDim.x;
    }
}
#endif