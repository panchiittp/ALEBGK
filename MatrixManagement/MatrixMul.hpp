#ifndef MATRIXMUL_HPP
#define MATRIXMUL_HPP
#include "CholeskyDecomposition.hpp"
#include "QRDecomposition.hpp"
#include "SORMethod.hpp"
__device__ __host__ void MatrixMatrixMul(double *A, double *B, double *C, int rowA,int colA,int rowB, int colB)
{
    for(int i=0;i<rowA;i++)
    {
        for(int j=0;j<colB;j++)
        {
            C[i*colB+j]=0.0;
            for(int k=0;k<colA;k++)
            {
                C[i*colB+j]+=A[i*colA+k]*B[k*colB+j];
            }
        }
    }
}

__device__ __host__ void MatrixVectorProduct(double *A, double *x, double *y, int rowA,int colA)
{
    for (int i = 0; i < rowA; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < colA; ++j) {
            y[i] += A[i * colA + j] * x[j];
        }
    }
}




__device__ void MatrixMulGPU(double *M,double *W,double *MTW, double *MTWM, double *MTWMInv, double *Id,double *MTWMInvMTW,int row, int col)
{
    for(int i=0;i<col;i++)
    {
        for(int j=0;j<row;j++)
        {
            MTW[i*row+j]=M[j*col+i]*W[j];
        }
    }

    MatrixMatrixMul(MTW,M,MTWM,col,row,row,col);
    // double *L=(double *)malloc(col*col * sizeof(double));
    
    // CholeskyDecomposition(MTWM,L,col);
    // double *LT=(double *)malloc(col*col * sizeof(double));
    // for(int i=0;i<col;i++)
    // {
    //     for(int j=0;j<col;j++)
    //     {
    //         LT[i*col+j]=L[j*col+i];
    //     }
    // }
    // MatrixMatrixMul(L,LT,MTWM,col,col,col,col);

    //InverseQR(MTWM,MTWMInv,col);
    InverseSOR(MTWM,MTWMInv,col);
    MatrixMatrixMul(MTWM,MTWMInv,Id,col,col,col,col);
    MatrixMatrixMul(MTWMInv,MTW,MTWMInvMTW,col,col,col,row);
}
#endif