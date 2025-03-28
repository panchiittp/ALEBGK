#ifndef NEIGHBOURSEARCH_HPP
#define NEIGHBOURSEARCH_HPP

__global__ void SortNeighbours(BGKParticle *dP, CalcParameters CalcParam)
{
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < CalcParam.N && dP[ind].active == true)
    {
        int n=dP[ind].totneigh;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (dP[ind].neighindex[j] > dP[ind].neighindex[j + 1]) {
                    // Swap arr[j] and arr[j+1]
                    int temp = dP[ind].neighindex[j];
                    dP[ind].neighindex[j] = dP[ind].neighindex[j + 1];
                    dP[ind].neighindex[j + 1] = temp;
                }
            }
        }
    }
}


void SortNeighboursCPU(BGKParticle *dP, CalcParameters CalcParam)
{
    for (int ind=0;ind < CalcParam.N;ind++)
    {
        if(dP[ind].active == true)
        {
            int n=dP[ind].totneigh;
            for (int i = 0; i < n - 1; i++) {
                for (int j = 0; j < n - i - 1; j++) {
                    if (dP[ind].neighindex[j] > dP[ind].neighindex[j + 1]) {
                        // Swap arr[j] and arr[j+1]
                        int temp = dP[ind].neighindex[j];
                        dP[ind].neighindex[j] = dP[ind].neighindex[j + 1];
                        dP[ind].neighindex[j + 1] = temp;
                    }
                }
            }
        }
    }
}

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
                if (pidx != -1 && dP[pidx].active == true)
                {
                    double distx = dP[i].pos.x - dP[pidx].pos.x;
                    double disty = dP[i].pos.y - dP[pidx].pos.y;

                    double dist = sqrt(distx * distx + disty * disty);
                    if (dist < CalcParam.radius)
                    {                          
                        dP[i].neighindex[neighIndexCount++] = pidx;                                        
                    }
                    // Compute corrected periodic distance
                    
                    if (dist > 3*CalcParam.radius && dP[pidx].boundary == false)
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


__device__ __host__ void IdentifyNeighbourTypeDevice(int p, BGKParticle *dP, double Lx,int k)
{
    //double dx[180],dy[180],dz[180];
    for (int i = 0; i < dP[p].totneigh; i++)
    {
        int neigh = dP[p].neighindex[i];
        double dx;
        if(k==0)
            dx = (dP[neigh].pos.x - dP[p].pos.x);
        if(k==1)
            dx = (dP[neigh].pos.y - dP[p].pos.y);
        if(k==2)
            dx = (dP[neigh].pos.z - dP[p].pos.z);            
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



void findNeighborParticlesPeriodicCPU(BGKParticle *dP, CalcParameters CalcParam, voxelDetails *voxinfo,DomainBoundary Domain)
{
    for(int i=0;i<CalcParam.N;i++)
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
        int count=0;
        for (int dy = -1; dy <= 1; ++dy)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                int nx = (x + dx + CalcParam.nbxBox) % CalcParam.nbxBox;
                int ny = (y + dy + CalcParam.nbyBox) % CalcParam.nbyBox;
    
                // Calculate the 1D index of the neighbor voxel
                int neighborIndex = nx + ny * CalcParam.nbxBox;
                dP[i].neighVoxel[neighVoxelCount++] = neighborIndex;
    
                count = voxinfo[neighborIndex].count;
                for (int j = 0; j < count; j++)
                {
                    
                    int pidx = voxinfo[neighborIndex].particleindex[j];
                    


                    if (pidx != -1 && dP[pidx].active == true)
                    {
                        double distx = dP[i].pos.x - dP[pidx].pos.x;
                        double disty = dP[i].pos.y - dP[pidx].pos.y;
                        

                        double dist = sqrt(distx * distx + disty * disty);
                        if (dist < CalcParam.radius)
                        {                          
                            dP[i].neighindex[neighIndexCount++] = pidx;         
                        
                               
                        }
                        // Compute corrected periodic distance
                        
                        if (dist > 3*CalcParam.radius && dP[pidx].boundary == false)
                        {
                            if (distx >=  Lx / 2.0) distx -= Lx;
                            if (distx <= -Lx / 2.0) distx += Lx;
    
                            if (disty >=  Ly / 2.0) disty -= Ly;
                            if (disty <= -Ly / 2.0) disty += Ly;
    
                            dist = sqrt(distx * distx + disty * disty);
                            if (dist < CalcParam.radius)
                            {
                                dP[i].neighindex[neighIndexCount++] = pidx;
                            }
                        }
                    }
                }                    
            }
        }
        dP[i].totvoxel = neighVoxelCount;
        dP[i].totneigh = neighIndexCount;
        std::cout<<"Completed the neighbour Search voxcount"<<count<<"Tot neigh"<<neighIndexCount<<std::endl;

    }
}


void IdentifyNeighbourTypeDeviceCPU(int p, BGKParticle *dP, double Lx,int k)
{
    //double dx[180],dy[180],dz[180];
    // std::cout<<"Total Number of Neighbours "<< dP[p].totneigh<<std::endl;

    for (int i = 0; i < dP[p].totneigh; i++)
    {
        int neigh = dP[p].neighindex[i];
        double dx;
        if(k==0)
            dx = (dP[neigh].pos.x - dP[p].pos.x);
        if(k==1)
            dx = (dP[neigh].pos.y - dP[p].pos.y);
        if(k==2)
            dx = (dP[neigh].pos.z - dP[p].pos.z);            
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
        // std::cout<<"Neighbour Type Identified "<<i<<std::endl;

    }
}
void IdentifyNeighbourTypeCPU(BGKParticle *dP,CalcParameters CalcParam,DomainBoundary Domain)
{    
    for(int p=0;p<CalcParam.N;p++)
    {
        // std::cout<<"Working on Neighbour Type Identification for "<<p<<std::endl;
        IdentifyNeighbourTypeDeviceCPU(p, dP,Domain.xright-Domain.xleft,0);
        IdentifyNeighbourTypeDeviceCPU(p, dP,Domain.ytop-Domain.ybottom,1);
    }
}


#endif