#ifndef NEIGHBOURSEARCH_HPP
#define NEIGHBOURSEARCH_HPP

__global__ void findNeighborParticlesPeriodic(BGKParticle *dP, CalcParameters CalcParam, voxelDetails *voxinfo,DomainBoundary Domain)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

if (i < CalcParam.N && dP[i].active == true)
{
    int voxIndex = dP[i].voxel;
    int x = voxIndex % CalcParam.nbxBox;
    int y = voxIndex / CalcParam.nbxBox;

    int neighVoxelCount = 0;
    int neighIndexCount = 0;
//    int neighTypeCount = 0;
    
    double Lx = (Domain.xright - Domain.xleft);
    double Ly = (Domain.ytop - Domain.ybottom);

    // Iterate over neighboring offsets in 2D
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            int nx = (x + dx + CalcParam.nbxBox) % CalcParam.nbxBox;
            int ny = (y + dy + CalcParam.nbyBox) % CalcParam.nbyBox;

            // Calculate the 1D index of the neighbor voxel
            int neighborIndex = nx + ny * CalcParam.nbxBox;
            dP[i].neighVoxel[neighVoxelCount++] = neighborIndex;

            int count = voxinfo[neighborIndex].count;
            for (int j = 0; j < count; j++)
            {
                int pidx = voxinfo[neighborIndex].particleindex[j];
                if (pidx != -1 && dP[pidx].active == 1)
                {
                    double distx = dP[i].x - dP[pidx].x;
                    double disty = dP[i].y - dP[pidx].y;

                    // Compute corrected periodic distance
                    double dist = sqrt(distx * distx + disty * disty);
                    if (dist < CalcParam.radius)
                    {                          
                        dP[i].neighindex[neighIndexCount++] = pidx;                                        
                    }
                    if (dist > 3*CalcParam.radius)// && dP[pidx].boundary == false)
                    {
                        if (distx >=  Lx / 2.0) distx -= Lx;
                        if (distx <= -Lx / 2.0) distx += Lx;

                        if (disty >=  Ly / 2.0) disty -= Ly;
                        if (disty <= -Ly / 2.0) disty += Ly;

                        dist = sqrt(distx * distx + disty * disty);
                        if (dist < CalcParam.radius)
                        {
                            dP[i].neighindex[neighIndexCount++] = pidx;
                            __syncthreads();
                        }
                    }
                }
            }                    
        }
    }
    dP[i].totvoxel = neighVoxelCount;
    dP[i].totneigh = neighIndexCount;
}

}


__device__ void IdentifyNeighbourTypeDevice(int p, BGKParticle *dP, double Lx,int k)
{
    //double dx[180],dy[180],dz[180];
    for (int i = 0; i < dP[p].totneigh; i++)
    {
        int neigh = dP[p].neighindex[i];
        double dx;
        if(k==0)
            dx = (dP[neigh].x - dP[p].x);
        if(k==1)
            dx = (dP[neigh].y - dP[p].y);
        if(k==2)
            dx = (dP[neigh].z - dP[p].z);            
        if(dx>Lx/2)
            dx=dx-Lx;
        else if (dx<-Lx/2)
            dx=dx+Lx;
        else
            dx=dx;
        
        if(dx>0)
        {
            dP[p].neightype[i*3+k]=k*2+1;
            dP[p].neighcount[k*2+1]++;
        }
        else if(dx<0)
        {
            dP[p].neightype[i*3+k]=k*2+0;
            dP[p].neighcount[k*2+0]++;
        }        
        else
        {
            dP[p].neightype[i*3+k]=-1;
        }                
    }
}

__global__ void IdentifyNeighbourType(BGKParticle *dP,CalcParameters CalcParam,DomainBoundary Domain)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p < CalcParam.N)
    {
        IdentifyNeighbourTypeDevice(p, dP,Domain.xright-Domain.xleft,0);
        IdentifyNeighbourTypeDevice(p, dP,Domain.ytop-Domain.ybottom,1);
    }
}

#endif