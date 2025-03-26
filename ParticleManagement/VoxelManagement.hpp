
#ifndef VOXELMANAGEMENT_HPP
#define VOXELMANAGEMENT_HPP

__device__ __host__ void Generatevoxel(voxelDetails *dvoxinfo,CalcParameters CalcParam,Parameters Param,DomainBoundary Domain)
{
    double x=Domain.xleft;
    double y=Domain.ybottom;
    double dx=CalcParam.xBox;
    double dy=CalcParam.yBox;
    double eps=1e-16; 
    for(int i=0;i<CalcParam.nbxBox;i++)
    {
        for(int j=0;j<CalcParam.nbyBox;j++)
        {
                int index=i+j*CalcParam.nbxBox;
                dvoxinfo[index].xmin=x+i*dx-eps;
                dvoxinfo[index].xmax=x+(i+1)*dx-eps;
                dvoxinfo[index].ymin=y+j*dy-eps;
                dvoxinfo[index].ymax=y+(j+1)*dy-eps;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
__device__ __host__ void VoxelNumber(BGKParticle &dP, CalcParameters CalcParam, DomainBoundary Domain,voxelDetails *voxinfo)
{




    // for(int i=0;i<CalcParam.nbxBox*CalcParam.nbyBox;i++)
    // {
    //     if(dP.x>=voxinfo[i].xmin && dP.x<voxinfo[i].xmax && dP.y>=voxinfo[i].ymin && dP.y<voxinfo[i].ymax)
    //     {
    //         dP.voxel=i;
    //         dP.z=voxinfo[i].xmin;
    //         dP.ux=voxinfo[i].xmax;
    //         dP.uy=voxinfo[i].ymin;
    //         dP.uz=voxinfo[i].ymax;
    //     }
    //     else if(dP.x==1 && dP.x>=voxinfo[i].xmin && dP.x<=voxinfo[i].xmax && dP.y>=voxinfo[i].ymin && dP.y<voxinfo[i].ymax )
    //     {
    //         dP.voxel=i;
    //         dP.z=voxinfo[i].xmin;
    //         dP.ux=voxinfo[i].xmax;
    //         dP.uy=voxinfo[i].ymin;
    //         dP.uz=voxinfo[i].ymax;
    //     }
    //     else if(dP.y==1 && dP.x>=voxinfo[i].xmin && dP.x<voxinfo[i].xmax && dP.y>=voxinfo[i].ymin && dP.y<=voxinfo[i].ymax )
    //     {
    //         dP.voxel=i;
    //         dP.z=voxinfo[i].xmin;
    //         dP.ux=voxinfo[i].xmax;
    //         dP.uy=voxinfo[i].ymin;
    //         dP.uz=voxinfo[i].ymax;
    //     }
    //     else if(dP.y==1 && dP.x==1 && dP.x>=voxinfo[i].xmin && dP.x<=voxinfo[i].xmax && dP.y>=voxinfo[i].ymin && dP.y<=voxinfo[i].ymax )
    //     {
    //         dP.voxel=i;
    //         dP.z=voxinfo[i].xmin;
    //         dP.ux=voxinfo[i].xmax;
    //         dP.uy=voxinfo[i].ymin;
    //         dP.uz=voxinfo[i].ymax;
    //     }

    // }

    // double xBoxSize = CalcParam.xBox;
    // double yBoxSize = CalcParam.yBox;
    // double eps = 1e-10;
    // // int xBox = (int)round((dP.x - Domain.xleft-eps) / xBoxSize);   // column index of voxel
    // // int yBox = (int)round((dP.y - Domain.ybottom-eps) / yBoxSize); // row index of voxel

    // int nbx = (int)floor((Domain.xright - Domain.xleft) / xBoxSize);
    // int nby = (int)floor((Domain.ytop - Domain.ybottom) / yBoxSize);
    


    // int xBox = min((int)floor((dP.x - Domain.xleft) / xBoxSize), nbx-1);
    // int yBox = min((int)floor((dP.y - Domain.ybottom) / yBoxSize), nby-1);
    // if (dP.x == Domain.xright)
    //     xBox = CalcParam.nbxBox - 1;
    // if (dP.y == Domain.ytop)
    //     yBox = CalcParam.nbyBox - 1;
    // //dP.voxel = xBox + CalcParam.nbxBox * yBox;
    // dP.voxel = xBox + yBox * nbx;
    // // return hBox + calcParam.nbhBox * vBox;
}

__device__ __host__ void VoxelInformation(BGKParticle &dP, CalcParameters CalcParam)
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

__global__ void GenerateVoxelNumberingKernel(BGKParticle *dP, CalcParameters CalcParam, DomainBoundary Domain, voxelDetails *voxinfo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < CalcParam.N)
    {
        //VoxelNumber(dP[i], CalcParam, Domain,voxinfo);
        int row = min(9,(int)(floor((dP[i].y + 1e-10)/ 0.1)));
        int col = min(9,(int)(floor((dP[i].x + 1e-10) / 0.1)));
    
        dP[i].voxel = row * 10 + col;
    }
}

__global__ void updateVoxelNumberingKernel(BGKParticle *dP, CalcParameters CalcParam, DomainBoundary Domain, voxelDetails *voxinfo)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < CalcParam.N)
    {
        int voxindex = dP[i].voxel;
        int count = atomicAdd(&voxinfo[voxindex].count, 1);
        __syncthreads();
        voxinfo[voxindex].particleindex[count] = i;
       
        VoxelInformation(dP[i], CalcParam);
        
        // i += blockDim.x * gridDim.x;
    }
}


void GenerateVoxelNumberingCPU(BGKParticle *dP, CalcParameters CalcParam, DomainBoundary Domain, voxelDetails *voxinfo)
{
    

    for(int i=0;i < CalcParam.N;i++)
    {
        //VoxelNumber(dP[i], CalcParam, Domain,voxinfo);
        int row = min(9,(int)(floor((dP[i].y + 1e-10)/ 0.1)));
        int col = min(9,(int)(floor((dP[i].x + 1e-10) / 0.1)));
    
        dP[i].voxel = row * 10 + col;
    }
}

void updateVoxelNumberingCPU(BGKParticle *dP, CalcParameters CalcParam, DomainBoundary Domain, voxelDetails *voxinfo)
{
    for(int i=0;i < CalcParam.N;i++)
    {
        //VoxelNumber(dP[i], CalcParam, Domain,voxinfo);
        int voxindex = dP[i].voxel;
        int count = voxinfo[voxindex].count++;
        voxinfo[voxindex].particleindex[count] = i;
        VoxelInformation(dP[i], CalcParam);
        // i += blockDim.x * gridDim.x;
    }
}
#endif