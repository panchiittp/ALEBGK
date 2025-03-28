#ifndef INITIALCONDITIONS_HPP
#define INITIALCONDITIONS_HPP
#define PI 3.14159265358979323846


__global__ void applyInitialConditionsKernel(BGKParticle *dP, CalcParameters CalcParam, Parameters Param, InitialConditions IC, Constants Constant)
{
    int particleIndex = blockIdx.x * blockDim.x + threadIdx.x;

    if (particleIndex < CalcParam.N)
    {
        double Ux = IC.Ux;
        double Uy = IC.Uy;
        double T = IC.T;
        double rho = IC.rho;
        dP[particleIndex].vel.x = Ux;
        dP[particleIndex].vel.y = Uy;
        dP[particleIndex].T = T;
        dP[particleIndex].rho = rho;
        if(dP[particleIndex].pos.y>=0.25 && dP[particleIndex].pos.y<=0.75)
        {
            dP[particleIndex].rho = 2;
            dP[particleIndex].T = 2.5/dP[particleIndex].rho;
            dP[particleIndex].vel.x = -0.5+0.01*sin(2*PI*dP[particleIndex].pos.x);//*sin(2*PI*dP[particleIndex].y);
            dP[particleIndex].vel.y = 0.01*sin(2*PI*dP[particleIndex].pos.x);//*sin(2*PI*dP[particleIndex].y);

        }        
        else
        {
            dP[particleIndex].rho = 1;
            dP[particleIndex].T = 2.5/dP[particleIndex].rho;
            dP[particleIndex].vel.x = 0.5+0.01*sin(2*PI*dP[particleIndex].pos.x);//*sin(2*PI*dP[particleIndex].y);
            dP[particleIndex].vel.y = 0.01*sin(2*PI*dP[particleIndex].pos.x);//*sin(2*PI*dP[particleIndex].y);
        }
        
            for (int j = 0; j < Param.Nv; ++j)
            {
                for (int i = 0; i < Param.Nv; ++i)
                {
                    int linearIndex = i + Param.Nv * j;
                    dP[particleIndex].g[linearIndex] = rho * std::exp(-(std::pow(CalcParam.vRange[i] - dP[particleIndex].vel.x, 2) + std::pow(CalcParam.vRange[j] - dP[particleIndex].vel.y, 2)) / (2 * Constant.R * T)) / std::pow((2 * Constant.R * 3.14159 * T), 1.5); // g1
                    dP[particleIndex].g[linearIndex + (Param.Nv * Param.Nv)] = Constant.R * T * dP[particleIndex].g[linearIndex]; //g2
                    //printf("gLinear Thread %d: dP[%d].rhs[%d] = %e\n", particleIndex, k, linearIndex, dP[particleIndex].g[linearIndex]);  
                    dP[particleIndex].gt[linearIndex] = dP[particleIndex].g[linearIndex];
                }
            }
    }
}


void applyInitialConditionsCPU(BGKParticle *dP, CalcParameters CalcParam, Parameters Param, InitialConditions IC, Constants Constant)
{


    for (int particleIndex=0;particleIndex < CalcParam.N;particleIndex++)
    {
        double Ux = IC.Ux;
        double Uy = IC.Uy;
        double T = IC.T;
        double rho = IC.rho;
        dP[particleIndex].vel.x = Ux;
        dP[particleIndex].vel.x = Uy;
        dP[particleIndex].T = T;
        dP[particleIndex].rho = rho;
        if(dP[particleIndex].pos.y>=0.25 && dP[particleIndex].pos.y<=0.75)
        {
            dP[particleIndex].rho = 2;
            dP[particleIndex].T = 2.5/dP[particleIndex].rho;
            dP[particleIndex].vel.x = -0.5+0.01*sin(2*PI*dP[particleIndex].pos.x);//*sin(2*PI*dP[particleIndex].y);
            dP[particleIndex].vel.y = 0.01*sin(2*PI*dP[particleIndex].pos.x);//*sin(2*PI*dP[particleIndex].y);

        }        
        else
        {
            dP[particleIndex].rho = 1;
            dP[particleIndex].T = 2.5/dP[particleIndex].rho;
            dP[particleIndex].vel.x = 0.5+0.01*sin(2*PI*dP[particleIndex].pos.x);//*sin(2*PI*dP[particleIndex].y);
            dP[particleIndex].vel.y = 0.01*sin(2*PI*dP[particleIndex].pos.x);//*sin(2*PI*dP[particleIndex].y);
        }
        // std::cout<<"Applying Initial Conditions for "<<particleIndex<<std::endl;
           for (int j = 0; j < Param.Nv; ++j)
            {
                for (int i = 0; i < Param.Nv; ++i)
                {
                    int linearIndex = i + Param.Nv * j;
                    dP[particleIndex].g[linearIndex] = rho * std::exp(-(std::pow(CalcParam.vRange[i] - dP[particleIndex].vel.x, 2) + std::pow(CalcParam.vRange[j] - dP[particleIndex].vel.y, 2)) / (2 * Constant.R * T)) / std::pow((2 * Constant.R * 3.14159 * T), 1.5); // g1
                    dP[particleIndex].g[linearIndex + (Param.Nv * Param.Nv)] = Constant.R * T * dP[particleIndex].g[linearIndex]; //g2
                    //printf("gLinear Thread %d: dP[%d].rhs[%d] = %e\n", particleIndex, k, linearIndex, dP[particleIndex].g[linearIndex]);  
                    dP[particleIndex].gt[linearIndex] = dP[particleIndex].g[linearIndex];
                }
            }
    }
}
#endif