#ifndef INTERPOLATIONCPU_HPP
#define INTERPOLATIONCPU_HPP
#include "FluxComputation.hpp"
#include "../MatrixManagement/MatrixMul.hpp"


void InverseJordanCPU(double *A,double *B,int n)
{
    double *A_copy = (double *)malloc(n*n * sizeof(double));
    for (int i = 0; i < n * n; i++)
        A_copy[i] = A[i];

    // Initialize B as the identity matrix
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            B[i * n + j] = (i == j) ? 1.0 : 0.0;

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < n; i++) {
        // Make the diagonal element 1
        double diag = A_copy[i * n + i];
        if (fabs(diag) < 1e-9) {
            free(A_copy);
            return;
        }

        for (int j = 0; j < n; j++) {
            A_copy[i * n + j] /= diag;
            B[i * n + j] /= diag;
        }

        // Make all other elements in the column 0
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = A_copy[k * n + i];
                for (int j = 0; j < n; j++) {
                    A_copy[k * n + j] -= factor * A_copy[i * n + j];
                    B[k * n + j] -= factor * B[i * n + j];
                }
            }
        }
    }
   
    #ifdef __CUDA_ARCH__
        __syncthreads();
    #endif    
    free(A_copy);
}


void MatrixMul(double *M,double *W,double *MTW, double *MTWM,double *MTWMInv, double *I, double *MTWMInvMTW,int *neightype, int row, int col)
{

    std::string direction[]={"Left","Right","Bottom","Top","Center","Top","Back"};
    std::string filename="mymatrix.dat";
    
    std::ofstream file;//(filename);
    // if (std::filesystem::exists(filename)) {
    //     std::filesystem::remove(filename);
    //     std::cout << "File deleted: " << filename << std::endl;
    // } else {
    //     std::cout << "File does not exist: " << filename << std::endl;
    // }
    file.open(filename,std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    file<<"Number of (Rows, Cols) = ("<<row<<","<<col<<")"<<std::endl;
    file<<"Neighbour Type"<<std::endl;
    for(int i=0;i<row;i++)
    {
            file<<direction[neightype[i]]<<" ";
    }
    file<<std::endl;
    file<<"M Matrix"<<std::endl;
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<col;j++)
        {
            file<<std::scientific<<M[i*col+j]<<" ";
        }
        file<<std::endl;
    }
    file<<"W Matrix"<<std::endl;
    for(int i=0;i<row;i++)
    {
        for(int j=0;j<row;j++)
        {
            if(i==j)
                file<<std::scientific<<W[i]<<" ";
            else
                file<<std::scientific<<0<<" ";
        }
        file<<std::endl;
    }
    for(int i=0;i<col;i++)
    {
        for(int j=0;j<row;j++)
        {
            MTW[i*row+j]=M[j*col+i]*W[j];
        }
    }

    // file<<"MTW Matrix"<<std::endl;
    // for(int i=0;i<col;i++)
    // {
    //     for(int j=0;j<row;j++)
    //     {
    //         file<<std::scientific<<MTW[i*row+j]<<" ";
    //     }
    //     file<<std::endl;
    // }
    for(int i=0;i<col;i++)
    {
        for(int j=0;j<col;j++)
        {
            MTWM[i*col+j]=0.0;
            for(int k=0;k<row;k++)
            {
                MTWM[i*col+j]+=MTW[i*row+k]*M[k*col+j];
            }
        }
    }


    //InverseJordanCPU(MTWM,MTWMInv,row);
    InverseSOR(MTWM,MTWMInv,col);
    // file<<"MTWM Matrix"<<std::endl;
    // for(int i=0;i<col;i++)
    // {
    //     for(int j=0;j<col;j++)
    //     {
    //         file<<std::scientific<<MTWM[i*col+j]<<" ";
    //     }
    //     file<<std::endl;
    // }

    // file<<"MTWM Inv Matrix"<<std::endl;
    // for(int i=0;i<col;i++)
    // {
    //     for(int j=0;j<col;j++)
    //     {
    //         file<<std::scientific<<MTWMInv[i*col+j]<<" ";
    //     }
    //     file<<std::endl;
    // }

    MatrixMatrixMul(MTWM,MTWMInv,I,col,col,col,col);
    MatrixMatrixMul(MTWMInv,MTW,MTWMInvMTW,col,col,col,row);
    file<<"Identity Matrix"<<std::endl;
    for(int i=0;i<col;i++)
    {
        for(int j=0;j<col;j++)
        {
            file<<std::scientific<<I[i*col+j]<<" ";
        }
        file<<std::endl;
    }

    file<<"MTWMINvMTW Matrix"<<std::endl;
    for(int i=0;i<col;i++)
    {
        for(int j=0;j<row;j++)
        {
            file<<std::scientific<<MTWMInvMTW[i*row+j]<<" ";
        }
        file<<std::endl;
    }

    file.close();

}

void CenterWENOCPU(int p, BGKParticle *dP, Parameters Param, CalcParameters CalcParam, Constants Constant,DomainBoundary Domain,int flag)
{
    int row;
    int fullrow=dP[p].totneigh;
    int col=6;
    if(flag==4)
        row=dP[p].totneigh;
    else
        row=dP[p].neighcount[flag];
    
    double Lx=(Domain.xright - Domain.xleft);
    double Ly=(Domain.ytop - Domain.ybottom);
    // double Lz=(Domain.zback - Domain.zfront);
    //printf("I am Inside this function\n");
    double *M=new double[row*6];
    double *W=new double[row];
    double *MTW=new double[6*row];
    double *MTWM=new double[6*6];
    int *neightype=new int[row];
    if(flag==4)
    {
        for (int i = 0; i < row; i++)
        {
            int neigh = dP[p].neighindex[i];
            double dx = (dP[neigh].pos.x - dP[p].pos.x);
            double dy = (dP[neigh].pos.y - dP[p].pos.y);
            // double dz = (dP[neigh].z - dP[p].z);
            double dist = std::sqrt(dx * dx + dy * dy);// + dz * dz);

            if(dist > 3*CalcParam.radius)
            {
                if (dx >=  Lx / 2.0) dx -= Lx;
                if (dx <= -Lx / 2.0) dx += Lx;

                if (dy >=  Ly / 2.0) dy -= Ly;
                if (dy <= -Ly / 2.0) dy += Ly;

                // if (dz >=  Lz / 2.0) dz -= Lz;
                // if (dz <= -Lz / 2.0) dz += Lz;
            }
            // //double dummy = std::pow(dx, 2) + std::pow(dy, 2);// + std::pow(dz, 2);
            // dist = sqrt(dx * dx + dy * dy);
            double weight=std::exp(-Constant.alpha * (dx*dx+dy*dy)/(CalcParam.radius*CalcParam.radius));
            
            M[i*col]=1;
            M[i*col+1]=dx;
            M[i*col+2]=0.5*dx*dx;
            M[i*col+3]=dy;
            M[i*col+4]=dx*dy;
            M[i*col+5]=0.5*dy*dy;
            W[i]=weight;
            neightype[i]=flag;

        }
    }
    else
    {
        int count=0;
        for (int i = 0; i < fullrow; i++)
        {
            int neigh = dP[p].neighindex[i];
            for(int k=3*i;k<3*(i+1);k++)
            {
                if(dP[p].neightype[k]==flag)
                {
                    double dx = (dP[neigh].pos.x - dP[p].pos.x);
                    double dy = (dP[neigh].pos.y - dP[p].pos.y);
                    // double dz = (dP[neigh].z - dP[p].z);
                    double dist = std::sqrt(dx * dx + dy * dy);// + dz * dz);

                    if(dist > 3*CalcParam.radius)
                    {
                        if (dx >=  Lx / 2.0) dx -= Lx;
                        if (dx <= -Lx / 2.0) dx += Lx;

                        if (dy >=  Ly / 2.0) dy -= Ly;
                        if (dy <= -Ly / 2.0) dy += Ly;

                        // if (dz >=  Lz / 2.0) dz -= Lz;
                        // if (dz <= -Lz / 2.0) dz += Lz;
                    }
                    // //double dummy = std::pow(dx, 2) + std::pow(dy, 2);// + std::pow(dz, 2);
                    // dist = sqrt(dx * dx + dy * dy);
                    double weight=std::exp(-Constant.alpha * (dx*dx+dy*dy)/(CalcParam.radius*CalcParam.radius));
                    
                    M[count*col]=1;
                    M[count*col+1]=dx;
                    M[count*col+2]=0.5*dx*dx;
                    M[count*col+3]=dy;
                    M[count*col+4]=dx*dy;
                    M[count*col+5]=0.5*dy*dy;
                    W[count]=weight;
                    neightype[count]=flag;
                    count++;
                }
        }
        }
    }
    double *Id=new double[col*col];
    double *MTWMInv=new double[col*col];
    double *MTWMInvMTW=new double[row*col];
    MatrixMul(M,W,MTW,MTWM,MTWMInv,Id,MTWMInvMTW,neightype,row,col);
    for(int i1=0;i1<col;i1++)
    {
        for(int j1=0;j1<row;j1++)
        {
            dP[p].MTW[i1*row+j1]=MTW[i1*row+j1];
        }
    }
    for(int i1=0;i1<6;i1++)
    {
        for(int j1=0;j1<6;j1++)
        {
            dP[p].MTWM[i1*6+j1]=MTWM[i1*6+j1];
        }
    }
    for(int i1=0;i1<6;i1++)
    {
        for(int j1=0;j1<6;j1++)
        {
            dP[p].MTWMInv[i1*6+j1]=MTWMInv[i1*6+j1];
        }
    }

    for(int i1=0;i1<6;i1++)
    {
        for(int j1=0;j1<6;j1++)
        {
            dP[p].Identity[i1*6+j1]=Id[i1*6+j1];
        }
    }
    for(int i1=0;i1<col;i1++)
    {
        for(int j1=0;j1<row;j1++)
        {
            dP[p].MTWMInvMTW[flag*6*50+i1*row+j1]=MTWMInvMTW[i1*row+j1];
        }
    }
    delete[] MTWM;

}

void ConstructCenterMMatrixCPU(BGKParticle *dP, Parameters Param, CalcParameters CalcParam, Constants Constant,DomainBoundary Domain,int flag)
{


    //for(int p=0;p<CalcParam.N;p++)
    int p=1251;
    {
        if (dP[p].boundary!=true)
        {
            for(int flag=0;flag<5;flag++)
                CenterWENOCPU(p, dP, Param, CalcParam, Constant,Domain,flag);
            
            for(int flag=0;flag<5;flag++)
                OptimizedFluxComputation(p,dP,Param,flag);

        }
    }
}
#endif