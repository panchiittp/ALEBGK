#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP
#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <chrono>
#include <limits>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <set>

////////////////////////////////////////////////////////////////////////////////////
//  Parameters
struct DomainBoundary
{
    double xleft, xright;
    double ytop, ybottom;
    double zfront, zback;
};

/* Velocity Boundary Conditiions in all three directions is defined using this structure */
struct UBc
{
    double Ux, Uy, Uz;
};

/* Boundary Conditions are Generated using this Structure*/
/* Wall Boundaries such as Left, Right etc
    Temperature (Tw), Density (rho)*/

struct BoundaryConditions
{
    double Tw = 270;
    double rho;
    // UBc Left, Right, Top, Bottom, Front, Back;
    double uxLeft = 0.0, uxRight = 0.0, uxTop = 100.0, uxBottom = 0.0, uxFront = 0.0, uxBack = 0.0;
    double uyLeft = 0.0, uyRight = 0.0, uyTop = 0.0, uyBottom = 0.0, uyFront = 0.0, uyBack = 0.0;
    double uzLeft = 0.0, uzRight = 0.0, uzTop = 0.0, uzBottom = 0.0, uzFront = 0.0, uzBack = 0.0;
};

/*Initial Conditions of U, T and rho*/

struct InitialConditions
{
    double Ux = 0.0, Uy = 0.0, Uz = 0.0;
    double T = 270, rho = 0.2203;
};

struct Constants
{
    double R = 208;
    double alpha = 6;
};

struct Parameters
{
    double VMax, VMin, d, tao, dt, tfinal, r, rb, radiusFactor;
    int Nv, Nx, Ny, Nz;
    int ndim;
    int saveFreq;
};

struct CalcParameters
{
    double dx, dy, dz;
    double dv,dt;
    double dv3; // cube of dv
    int N;
    double xBox, yBox, zBox; // Sizes of voxels
    int nbxBox, nbyBox, nbzBox;
    int nvox;
    double radius;
    double minDist;
    double vRange[31];
};
__constant__ CalcParameters GPUCalcParam;


#include "ReadParameters.hpp"
#include "CalculateParameters.hpp"
#endif