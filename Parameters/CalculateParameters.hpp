#ifndef CALCULATEPARAMETERS_H
#define CALCULATEPARAMETERS_H

/*This function calculates the parameters such as
dx - distance between adjacent particles in x direction
dy - distance between adjacent particles in y direction
dz - distance between adjacent particles in z direction
N - Number of particles*/

void CalculateParameters()
{
    CalcParam.dx = (Domain.xright - Domain.xleft) / (double)(Param.Nx - 1);
    CalcParam.dy = (Domain.ytop - Domain.ybottom) / (double)(Param.Ny - 1);
    if(Param.ndim==3)
        CalcParam.dz = (Domain.zback - Domain.zfront) / (double)(Param.Nz - 1);    //For 3D
    else
        CalcParam.dz = 1;
    CalcParam.N = Param.Nx * Param.Ny * Param.Nz;
    CalcParam.xBox = CalcParam.dx * Param.radiusFactor;
    CalcParam.yBox = CalcParam.dy * Param.radiusFactor;
    CalcParam.zBox = CalcParam.dz * Param.radiusFactor;
    CalcParam.radius = CalcParam.dx * Param.radiusFactor;
    CalcParam.minDist = CalcParam.dx * 0.05;

    CalcParam.nbxBox = ceil((Domain.xright - Domain.xleft) / CalcParam.xBox); // Amount of voxels in horizontal direction
    CalcParam.nbyBox = ceil((Domain.ytop - Domain.ybottom) / CalcParam.yBox); // Amount of voxels in vertical direction
    if(Param.ndim==3)
        CalcParam.nbzBox = ceil((Domain.zback - Domain.zfront) / CalcParam.zBox); // Amount of voxels in front-back direction
    else
        CalcParam.nbzBox = 1;    
    CalcParam.nvox = CalcParam.nbxBox * CalcParam.nbyBox * CalcParam.nbzBox;
    CalcParam.dt=Param.dt;
    for (int i = 1; i < Param.Nv; i++)
    {
        CalcParam.vRange[i] = CalcParam.vRange[i - 1] + CalcParam.dv;
    }
    CalcParam.vRange[Param.Nv - 1] = Param.VMax;
}
#endif