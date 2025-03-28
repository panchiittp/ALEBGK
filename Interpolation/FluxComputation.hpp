#ifndef FLUXCOMPUTATION_HPP
#define FLUXCOMPUTATION_HPP
__device__ __host__ void MatrixVecMul(int p, BGKParticle *dP,int flag,int gt)
{
    int row;
    if(flag==4)
    {
        row= dP[p].totneigh;
    }
    else
    {
        row=dP[p].neighcount[flag];
    }

    int col=6;
    //int N=10;

    for(int i=0;i<col;i++)
    {
        dP[p].gWENO[flag*col+i]=0.0;
        dP[p].gWENO2[flag*col+i]=0.0;
    }

    for(int i=0;i<col;i++)
    {
        for(int j=0;j<row;j++)
        {
            if(gt==1)
                dP[p].gWENO[flag*col+i]+=1;//dP[p].MTWMInvMTW[flag*6*50+i*row+j]*dP[p].rhs[j];
            else
                dP[p].gWENO2[flag*col+i]+=2;//dP[p].MTWMInvMTW[flag*6*50+i*row+j]*dP[p].rhs[j];
        }
    }
}


__device__ __host__ void OptimizedFluxComputation(int p, BGKParticle *dP,Parameters Param,int flag)
{
    int row;
    int fullrow=dP[p].totneigh;
    if(flag==4)
    {
        row= dP[p].totneigh;
        for (int k = 0; k < Param.Nv; k++)
        {
            for (int j = 0; j < Param.Nv; j++)
            {
                int linearIndex = j + Param.Nv * k;
                for (int i1 = 0; i1 < row; i1++)
                {                
                    dP[p].rhs[i1]=dP[dP[p].neighindex[i1]].g[linearIndex] - dP[p].g[linearIndex];                    
                }
                MatrixVecMul(p,dP,flag,1);

                for (int i1 = 0; i1 < row; i1++)
                {                
                    dP[p].rhs[i1]=dP[dP[p].neighindex[i1]].g[linearIndex + (Param.Nv * Param.Nv)] - dP[p].g[linearIndex + (Param.Nv * Param.Nv)];                    
                }
                MatrixVecMul(p,dP,flag,2);
            }
        }
    }
    else
    {
        row=dP[p].neighcount[flag];
        for (int k = 0; k < Param.Nv; k++)
        {
            for (int j = 0; j < Param.Nv; j++)
            {
                int linearIndex = j + Param.Nv * k;
                int count=0;
                for (int i1 = 0; i1 < fullrow; i1++)
                {                
                    for(int k=3*i1;k<3*(i1+1);k++)
                    {
                        if(dP[p].neightype[k]==flag)                        
                        {
                            dP[p].rhs[count]=dP[dP[p].neighindex[i1]].g[linearIndex] - dP[p].g[linearIndex];
                            count++;
                        }
                        
                    }
                }
                MatrixVecMul(p,dP,flag,1);
                count=0;
                for (int i1 = 0; i1 < fullrow; i1++)
                {                
                    for(int k=3*i1;k<3*(i1+1);k++)
                    {
                        if(dP[p].neightype[k]==flag)                        
                        {
                            dP[p].rhs[count]=dP[dP[p].neighindex[i1]].g[linearIndex + (Param.Nv * Param.Nv)] - dP[p].g[linearIndex + (Param.Nv * Param.Nv)];
                            count++;
                        }
                        
                    }
                }
                MatrixVecMul(p,dP,flag,2);
            }
        }
    }


}
#endif
