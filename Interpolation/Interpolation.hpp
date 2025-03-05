#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

__device__ int index(int i, int j,int N)
{
    if(j>=i)
    {
        return N*i+j-i*(i+1)/2;
    }
    else
    {
        return N*j+i-j*(j+1)/2;
    }
}
__device__ void SymmetricInverseJordan(int p,BGKParticle *dP)
{
    int N=6;
    double A[6][6];
    double invA[6][6];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = dP[p].MTWM[index(i,j,N)];
        }
    }
    __syncthreads();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            invA[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    __syncthreads();
    // Perform Gauss-Jordan elimination
    for (int i = 0; i < N; i++) {
        double diag = A[i][i]; // Pivot element

        // Normalize the pivot row
        for (int j = 0; j < N; j++) {
            A[i][j] /= diag;
            invA[i][j] /= diag;
        }

        // Make all other elements in column zero
        for (int j = 0; j < N; j++) {
            if (j != i) {
                double factor = A[j][i];
                for (int k = 0; k < N; k++) {
                    A[j][k] -= factor * A[i][k];
                    invA[j][k] -= factor * invA[i][k];
                }
            }
        }
    }
    __syncthreads();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dP[p].MTWM[index(i,j,N)]=invA[i][j];
        }
    }
    __syncthreads();
}

// __device__ void SymmetricInverseCholesky(int p, BGKParticle *dP)
// {
//     int n=10;
//     // for(int i=0;i<55;i++)
//     // {
//     //     dP[p].MTWMInv[i]=0;
//     // }
//     // for(int i=0;i<n;i++)
//     // {
//     //     dP[p].MTWMInv[index(i,i,n)]=1.0;
//     // }

//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j <= i; ++j) {
//             double sum = 0.0;

//             for (int k = 0; k < j; ++k) {
//                 sum += dP[p].MTWMInv[index(i, k,n)] * dP[p].MTWMInv[index(j, k,n)];
//             }

//             if (i == j) {
//                 dP[p].MTWMInv[index(i, j,n)] = std::sqrt(dP[p].MTWM[index(i, i,n)] - sum);
//             } else {
//                 dP[p].MTWMInv[index(i, j,n)] = (dP[p].MTWM[index(i, j,n)] - sum) /dP[p].MTWMInv[index(j, j,n)];
//             }
//         }
//     }
//     double L[55];
//     for(int i=0;i<55;i++)
//     {
//         L[i]=dP[p].MTWMInv[i];
//         dP[p].MTWM[i]=dP[p].MTWMInv[i];
//         dP[p].MTWMInv[i]=0.0;

//     }
//     //Forward Substititution
//     double Y[55];
//     // double x[10],y[10],b[10];
//     // for(int i=0;i<10;i++)
//     // {
//     //     b[i]=0.0;
//     // }
//     // b[0]=1.0;
//     // y[0]=b[9]/dP[p].MTWMInv[0];
//     // for(int i=1;i<n;i++)
//     // {
//     //     double sum=b[i];
//     //     for(int j=0;j<i;j++)
//     //     {
//     //         sum=sum-dP[p].MTWMInv[index(i,j,n)]*y[j];
//     //     }
//     //     y[i]=sum/dP[p].MTWMInv[index(i,i,n)];
//     // }
//     // x[9]=y[9]/y[55];
//     // for(int i=n-1;i>=0;i--)
//     // {
//     //     double sum=y[i];
//     //     for(int j=i+1;j<n;j++)
//     //     {
//     //         sum=sum-L[index(j,i,n)]*x[j];
//     //     }
//     //     x[i]=sum/L[index(i,i,n)];

//     // }
//     //double y[10];
//     for (int i = 0; i < n; ++i) {
//         for (int j = 0; j < n; ++j) {
//             double sum = 0.0;

//             for (int k = 0; k < i; ++k) {
//                 sum +=L[index(i, k, n)] * Y[index(k, j, n)];
//             }
//             if(i==j)
//                 Y[index(i, j,n)] = (1-sum)/L[index(i, i,n)];// (I[index(i,j,n)] - sum) / L[index(i, i,n)];
//             else
//                 Y[index(i, j,n)] = -sum/L[index(i, i,n)];
//         }
//     }
//     // __syncthreads();
//     // for(int i=0;i<55;i++)
//     // {
//     //     dP[p].MTWMInv[i]=Y[i];
//     //     // if(i<10)
//     //     //     dP[p].MTWMInv[i]=Y[i];
//     // }
//     //Backward Substitution
//     for (int i = n - 1; i >= 0; --i) {
//         for (int j = 0; j < n; ++j) {
//             double sum = 0.0;

//             for (int k = i + 1; k < n; ++k) {
//                 sum += L[index(k, i,n)] * dP[p].MTWMInv[index(k, j,n)];
//             }

//             dP[p].MTWMInv[index(i, j,n)] = (Y[index(i, j,n)] - sum) / L[index(i, i,n)];
//         }
//     }
// }


__device__ void MatrixMatrixMul(int p, BGKParticle *dP,int flag)
{
    int row;
    if(flag==6)
    {
        row= dP[p].totneigh;
    }
    else
    {
        row=dP[p].neighcount[flag];
    }
    int col=6;
    double C[6*250];
    for(int i=0;i<row*col;i++)
    {
        C[i]=0.0;
    }
    for (int i = 0; i < col; ++i) {         // Loop over rows of A
        for (int j = 0; j < row; ++j) {     // Loop over columns of BT
            for (int k = 0; k < col; ++k) { // Loop over common dimension
                C[i * row + j] += dP[p].MTWM[index(i,k,col)] * dP[p].MTW[k * row + j];
            }
        }
    }
    __syncthreads();
    for(int i=0;i<row*col;i++)
    {
        dP[p].MTW[i]=C[i];
    }
    __syncthreads();

}


__device__ void MatrixVecMul(int p, BGKParticle *dP,int flag)
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
        for(int j=0;j<row;j++)
        {
            dP[p].gWENO[flag*10+i]+=dP[p].MTW[i*row+j]*dP[p].rhs[j];
        }
    }
}
    // int row = dP[p].totneigh;
    // int col = 10;

    // int i = threadIdx.y; // Row index
    // int j = threadIdx.x; // Column index

    // if (i < col && j < row) {
    //     double val = dP[p].MTW[i * row + j] * dP[p].rhs[j];
    //     atomicAdd(&dP[p].gWENO[i], val);
    // }
// }


__device__ void OptimizedFluxComputation(int p, BGKParticle *dP,Parameters Param,int flag)
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

    for (int k = 0; k < Param.Nv; k++)
    {
        for (int j = 0; j < Param.Nv; j++)
        {
            int linearIndex = j + Param.Nv * k;
            for (int i1 = 0; i1 < row; i1++)
            {                
                if(dP[p].neightype[i1]==flag)
                    dP[p].rhs[i1]=dP[dP[p].neighindex[i1]].g[linearIndex] - dP[p].g[linearIndex];
                // if(dP[p].neightype[i1]==4)
                //     dP[p].rhs[i1]=dP[dP[p].neighindex[i1]].g[linearIndex] - dP[p].g[linearIndex];
            }
            MatrixVecMul(p,dP,flag);
        }
    }
}


__device__ void ComputeOtherMTWM(int p, BGKParticle *dP, Parameters Param, CalcParameters CalcParam, Constants Constant,DomainBoundary Domain,int flag)
{
    int row=dP[p].neighcount[flag];
    dP[p].checkvariable=row;
    double Lx=(Domain.xright - Domain.xleft);
    double Ly=(Domain.ytop - Domain.ybottom);
    // double Lz=(Domain.zback - Domain.zfront);
    for (int i = 0; i < dP[p].totneigh; i++)
    {
        if(dP[p].neightype[i]==flag)
        {
            int neigh = dP[p].neighindex[i];
            double dx = (dP[neigh].x - dP[p].x);
            double dy = (dP[neigh].y - dP[p].y);
            // double dz = (dP[neigh].z - dP[p].z);
            double dist = sqrt(dx * dx + dy * dy);// + dz * dz);

            if(dist > 3*CalcParam.radius)
            {
                if (dx >=  Lx / 2.0) dx -= Lx;
                if (dx <= -Lx / 2.0) dx += Lx;

                if (dy >=  Ly / 2.0) dy -= Ly;
                if (dy <= -Ly / 2.0) dy += Ly;

                // if (dz >=  Lz / 2.0) dz -= Lz;
                // if (dz <= -Lz / 2.0) dz += Lz;
            }
            double dummy = std::pow(dx, 2) + std::pow(dy, 2);// + std::pow(dz, 2);
            double weight=std::exp(-Constant.alpha * (dummy)/(CalcParam.radius*CalcParam.radius));
            // dP[p].W[i] = weight;
            dP[p].MTW[i]=1.0*weight;
            dP[p].MTW[row+i]=dx*weight;
            dP[p].MTW[2*row+i]=0.5*dx*dx*weight;
            dP[p].MTW[3*row+i]=dy*weight;
            dP[p].MTW[4*row+i]=dx*dy*weight;
            dP[p].MTW[5*row+i]=0.5*dy*dy*weight;
            // dP[p].MTW[6*row+i]=dz*weight;
            // dP[p].MTW[7*row+i]=dx*dz*weight;
            // dP[p].MTW[8*row+i]=dy*dz*weight;
            // dP[p].MTW[9*row+i]=0.5*dz*dz*weight;


            dP[p].MTWM[0]+=weight; //(0,0)
            dP[p].MTWM[1]+=weight*dx; //(0,1)
            dP[p].MTWM[2]+=weight*dx*dx*0.5; //(0,2)
            dP[p].MTWM[3]+=weight*dy; //(0,3)
            dP[p].MTWM[4]+=weight*dx*dy; //(0,4)
            dP[p].MTWM[5]+=weight*dy*dy*0.5; //(0,5)
            // dP[p].MTWM[6]+=weight*dz;//(0,6)
            // dP[p].MTWM[7]+=weight*dx*dz;//(0,7)
            // dP[p].MTWM[8]+=weight*dy*dz;
            // dP[p].MTWM[9]+=weight*dz*dz*0.5;

            dP[p].MTWM[6]+=weight*dx*dx;
            dP[p].MTWM[7]+=weight*dx*dx*dx*0.5;
            dP[p].MTWM[8]+=weight*dy*dx;
            dP[p].MTWM[9]+=weight*dx*dy*dx;
            dP[p].MTWM[10]+=weight*dy*dy*dx*0.5;
            // dP[p].MTWM[15]+=weight*dz*dx;
            // dP[p].MTWM[16]+=weight*dx*dz*dx;
            // dP[p].MTWM[17]+=weight*dy*dz*dx;
            // dP[p].MTWM[18]+=weight*dz*dz*dx*0.5;

            dP[p].MTWM[11]+=weight*dx*dx*0.5*dx*dx*0.5;
            dP[p].MTWM[12]+=weight*dy*dx*dx*0.5;
            dP[p].MTWM[13]+=weight*dx*dy*dx*dx*0.5;
            dP[p].MTWM[14]+=weight*dy*dy*0.5*dx*dx*0.5;
            // dP[p].MTWM[15]+=weight*dz*dx*dx*0.5;
            // dP[p].MTWM[24]+=weight*dx*dz*dx*dx*0.5;
            // dP[p].MTWM[25]+=weight*dy*dz*dx*dx*0.5;
            // dP[p].MTWM[26]+=weight*dz*dz*0.5*dx*dx*0.5;


            dP[p].MTWM[15]+=weight*dy*dy;
            dP[p].MTWM[16]+=weight*dx*dy*dy;
            dP[p].MTWM[17]+=weight*dy*dy*dy*0.5;
            // dP[p].MTWM[30]+=weight*dz*dy;
            // dP[p].MTWM[31]+=weight*dx*dz*dy;
            // dP[p].MTWM[32]+=weight*dy*dz*dy;
            // dP[p].MTWM[33]+=weight*dz*dz*dy*0.5;

            dP[p].MTWM[18]+=weight*dx*dy*dy*dx;
            dP[p].MTWM[19]+=weight*dy*dy*dy*dx*0.5;
            // dP[p].MTWM[21]+=weight*dz*dy*dx;
            // dP[p].MTWM[37]+=weight*dx*dz*dy*dx;
            // dP[p].MTWM[38]+=weight*dy*dz*dy*dx;
            // dP[p].MTWM[39]+=weight*dz*dz*dy*0.5*dx;


            dP[p].MTWM[20]+=weight*dy*dy*dy*dy*0.5*0.5;
            // dP[p].MTWM[23]+=weight*dz*dy*dy*0.5;
            // dP[p].MTWM[42]+=weight*dx*dz*dy*dy*0.5;
            // dP[p].MTWM[43]+=weight*dy*dz*dy*dy*0.5;
            // dP[p].MTWM[44]+=weight*dz*dz*dy*dy*0.5*0.5;

            // dP[p].MTWM[45]+=weight*dz*dz;
            // dP[p].MTWM[46]+=weight*dx*dz*dz;
            // dP[p].MTWM[47]+=weight*dy*dz*dz;
            // dP[p].MTWM[48]+=weight*dz*dz*dz*0.5;

            // dP[p].MTWM[49]+=weight*dx*dz*dx*dz;
            // dP[p].MTWM[50]+=weight*dy*dz*dx*dz;
            // dP[p].MTWM[51]+=weight*dz*dz*dx*dz*0.5;

            // dP[p].MTWM[52]+=weight*dy*dz*dy*dz;
            // dP[p].MTWM[53]+=weight*dz*dz*dy*dz*0.5;

            // dP[p].MTWM[54]+=weight*dz*dz*dz*dz*0.5*0.5;
        }
    }
}

__device__ void ComputeCenterMTWM(int p, BGKParticle *dP, Parameters Param, CalcParameters CalcParam, Constants Constant,DomainBoundary Domain)
{
    int row=dP[p].totneigh;
    double Lx=(Domain.xright - Domain.xleft);
    double Ly=(Domain.ytop - Domain.ybottom);
    // double Lz=(Domain.zback - Domain.zfront);
    //printf("I am Inside this function\n");
    for (int i = 0; i < dP[p].totneigh; i++)
    {
        int neigh = dP[p].neighindex[i];
        double dx = (dP[neigh].x - dP[p].x);
        double dy = (dP[neigh].y - dP[p].y);
        // double dz = (dP[neigh].z - dP[p].z);
        double dist = sqrt(dx * dx + dy * dy);// + dz * dz);

        if(dist > 3*CalcParam.radius)
        {
            if (dx >=  Lx / 2.0) dx -= Lx;
            if (dx <= -Lx / 2.0) dx += Lx;

            if (dy >=  Ly / 2.0) dy -= Ly;
            if (dy <= -Ly / 2.0) dy += Ly;

            // if (dz >=  Lz / 2.0) dz -= Lz;
            // if (dz <= -Lz / 2.0) dz += Lz;
        }
        double dummy = std::pow(dx, 2) + std::pow(dy, 2);// + std::pow(dz, 2);
        double weight=std::exp(-Constant.alpha * (dummy)/(CalcParam.radius*CalcParam.radius));
        // dP[p].W[i] = weight;
        dP[p].MTW[i]=1.0*weight;
        dP[p].MTW[row+i]=dx*weight;
        dP[p].MTW[2*row+i]=0.5*dx*dx*weight;
        dP[p].MTW[3*row+i]=dy*weight;
        dP[p].MTW[4*row+i]=dx*dy*weight;
        dP[p].MTW[5*row+i]=0.5*dy*dy*weight;
        // dP[p].MTW[6*row+i]=dz*weight;
        // dP[p].MTW[7*row+i]=dx*dz*weight;
        // dP[p].MTW[8*row+i]=dy*dz*weight;
        // dP[p].MTW[9*row+i]=0.5*dz*dz*weight;



        dP[p].MTWM[0]+=weight; //(0,0)
        dP[p].MTWM[1]+=weight*dx; //(0,1)
        dP[p].MTWM[2]+=weight*dx*dx*0.5; //(0,2)
        dP[p].MTWM[3]+=weight*dy; //(0,3)
        dP[p].MTWM[4]+=weight*dx*dy; //(0,4)
        dP[p].MTWM[5]+=weight*dy*dy*0.5; //(0,5)
        // dP[p].MTWM[6]+=weight*dz;//(0,6)
        // dP[p].MTWM[7]+=weight*dx*dz;//(0,7)
        // dP[p].MTWM[8]+=weight*dy*dz;
        // dP[p].MTWM[9]+=weight*dz*dz*0.5;

        dP[p].MTWM[6]+=weight*dx*dx;
        dP[p].MTWM[7]+=weight*dx*dx*dx*0.5;
        dP[p].MTWM[8]+=weight*dy*dx;
        dP[p].MTWM[9]+=weight*dx*dy*dx;
        dP[p].MTWM[10]+=weight*dy*dy*dx*0.5;
        // dP[p].MTWM[15]+=weight*dz*dx;
        // dP[p].MTWM[16]+=weight*dx*dz*dx;
        // dP[p].MTWM[17]+=weight*dy*dz*dx;
        // dP[p].MTWM[18]+=weight*dz*dz*dx*0.5;

        dP[p].MTWM[11]+=weight*dx*dx*0.5*dx*dx*0.5;
        dP[p].MTWM[12]+=weight*dy*dx*dx*0.5;
        dP[p].MTWM[13]+=weight*dx*dy*dx*dx*0.5;
        dP[p].MTWM[14]+=weight*dy*dy*0.5*dx*dx*0.5;
        // dP[p].MTWM[15]+=weight*dz*dx*dx*0.5;
        // dP[p].MTWM[24]+=weight*dx*dz*dx*dx*0.5;
        // dP[p].MTWM[25]+=weight*dy*dz*dx*dx*0.5;
        // dP[p].MTWM[26]+=weight*dz*dz*0.5*dx*dx*0.5;


        dP[p].MTWM[15]+=weight*dy*dy;
        dP[p].MTWM[16]+=weight*dx*dy*dy;
        dP[p].MTWM[17]+=weight*dy*dy*dy*0.5;
        // dP[p].MTWM[30]+=weight*dz*dy;
        // dP[p].MTWM[31]+=weight*dx*dz*dy;
        // dP[p].MTWM[32]+=weight*dy*dz*dy;
        // dP[p].MTWM[33]+=weight*dz*dz*dy*0.5;

        dP[p].MTWM[18]+=weight*dx*dy*dy*dx;
        dP[p].MTWM[19]+=weight*dy*dy*dy*dx*0.5;
        // dP[p].MTWM[21]+=weight*dz*dy*dx;
        // dP[p].MTWM[37]+=weight*dx*dz*dy*dx;
        // dP[p].MTWM[38]+=weight*dy*dz*dy*dx;
        // dP[p].MTWM[39]+=weight*dz*dz*dy*0.5*dx;


        dP[p].MTWM[20]+=weight*dy*dy*dy*dy*0.5*0.5;
        // dP[p].MTWM[23]+=weight*dz*dy*dy*0.5;
        // dP[p].MTWM[42]+=weight*dx*dz*dy*dy*0.5;
        // dP[p].MTWM[43]+=weight*dy*dz*dy*dy*0.5;
        // dP[p].MTWM[44]+=weight*dz*dz*dy*dy*0.5*0.5;

        // dP[p].MTWM[45]+=weight*dz*dz;
        // dP[p].MTWM[46]+=weight*dx*dz*dz;
        // dP[p].MTWM[47]+=weight*dy*dz*dz;
        // dP[p].MTWM[48]+=weight*dz*dz*dz*0.5;

        // dP[p].MTWM[49]+=weight*dx*dz*dx*dz;
        // dP[p].MTWM[50]+=weight*dy*dz*dx*dz;
        // dP[p].MTWM[51]+=weight*dz*dz*dx*dz*0.5;

        // dP[p].MTWM[52]+=weight*dy*dz*dy*dz;
        // dP[p].MTWM[53]+=weight*dz*dz*dy*dz*0.5;

        // dP[p].MTWM[54]+=weight*dz*dz*dz*dz*0.5*0.5;
    }
}
__device__ void CenterWENO(int p, BGKParticle *dP, Parameters Param, CalcParameters CalcParam, Constants Constant,DomainBoundary Domain,int flag)
{
    // ComputeCenterMTWM(p, dP, Param, CalcParam, Constant,Domain);

    if(flag==4)
    {
        ComputeCenterMTWM(p, dP, Param, CalcParam, Constant,Domain);
        __syncthreads();

    }
    else
    {
        ComputeOtherMTWM(p, dP, Param, CalcParam, Constant,Domain,flag);
        __syncthreads();

    }

    SymmetricInverseJordan(p,dP);
        __syncthreads();
    MatrixMatrixMul(p,dP,flag);
     __syncthreads();
    //for (int l = 0; l < Param.Nv; l++)
    OptimizedFluxComputation(p,dP,Param,flag);
    __syncthreads();

    //
    // {
    //     for(int j=0;j<col;j++)
    //     {
    //         dP[p].M[i*col+j]=M[i*col+j];
    //     }
    //     dP[p].W[i]=W[i];
    // }
}



__global__ void ConstructCenterMMatrixKernel(BGKParticle *dP, Parameters Param, CalcParameters CalcParam, Constants Constant,DomainBoundary Domain,int flag)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p < CalcParam.N && dP[p].boundary!=true)
    {
            CenterWENO(p, dP, Param, CalcParam, Constant,Domain,flag);

    }
}
#endif