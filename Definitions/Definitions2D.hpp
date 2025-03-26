#ifndef DEFINITIONS_HPP
#define DEFINITIONS_HPP
#include<iostream>
//#include<cuda_runtime.h>
//#include<device_launch_parameters.h>
#include<string>
#include<cmath>
#include<cstring>
#include<fstream>
#include<set>
#include<iomanip>
#include <cassert>
#include <stdexcept>
// #include "ProblemDefinitions.hpp"
// #include "InitialandBoundaryConditions.hpp"
// #include "VectorDefinitions.hpp"
// #include "MatrixDefinitions.hpp"
// #include "LinearSystem.hpp"
struct Properties
{
    double k; //k is specific heat constant
    double rho; //rho - density
    double cp,nu; //nu - kinmetatic viscosity, cp-thermal conductivity
    double T,Told,p,pold; //T-temperature, p-pressure
    double mu, kappa; //dynamic viscosity, kappa - heat coefficient
};

struct BGKParticle
{
    double x, y, z, ux, uy, uz, T, rho;
    bool boundary, validg, active;
    int voxel, totvoxel, totneigh; // Linear index of the voxel in the 2D grid in which the particle is residing
    int neighVoxel[27];            // neighbouring voxels
    int neighindex[50];
    int neightype[3*50];   //Each particle can be a neighbour of a particle in only 2 ways (maximum).
    double M[6*50];
    double W[50];
    double MTW[6*50];
    double MTWM[21];
    double gWENO[5*6]; //First 6 for Center, next 6 for Left, then Right, Bottom, Top, Front, Back.
    double rhs[50];
    int neighcount[6];
    int checkvariable;
//    Point nor[180], tgt[180], binor[180];//,wdiff[180]; //dx,dy,dz, Normal: nx,ny,nz,Tangent: tx,ty,tz, Binormal: bx,by,bz, wdx,wdy,wdz
//    Angle anglebar[180];
    
    // int active;
    //  int active;  // added by me to check whether particle removed or not, 1 implies not removed, 0 implies removed

    double g[2000];  // contains g1 and g2 row-wise vectorized
    double gt[2000]; // d tilda
                      // double maxwellian[1000]; //maxwellian vectorised
    };

struct voxelDetails
{
    int count;
    int particleindex[100];
    double xmin,xmax,ymin,ymax,zmin,zmax;
    __host__ __device__ voxelDetails() : count(0)
    {
        for (int i = 0; i < 100; i++)
        {
            particleindex[i] = -1;
        }
    }
};

struct CSRMatrix
{
    int numrow,numcol,nnz;
    int *numPtr,*numCols;
    double *vals;
}

#endif