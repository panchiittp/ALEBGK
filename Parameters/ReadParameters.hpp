#ifndef READPARAMETERS_H
#define READPARAMETERS_H
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include "nlohmann/json.hpp"
using namespace std;
using json = nlohmann::json;

/* JSON Files are used for reading the parameters from a json file
   Domain Boundary, Boundary Condition, Initial Condition, Constants andd Parameters are initialized here
   Calculated parameters such as dx, dy, dz, Number of Particles (N), dv are computed
*/
DomainBoundary Domain;
BoundaryConditions BC;
InitialConditions IC;
Constants Constant;
Parameters Param;
CalcParameters CalcParam;

/* This function reads the parameters from the parameters.json file*/
void readparameters()
{

    std::ifstream file("parameters.json");

    // Check if the file is open successfully
    if (!file.is_open())
    {
        std::cerr << "Error opening the file!" << std::endl;
        return; // Return an error code
    }
    else
    {
        std::cout<<"File Reading is Successfull"<<std::endl;
    }
    // Read the content of the file into a JSON object
    json jsonData;
    file >> jsonData;

    // Close the file
    file.close();
    Param.ndim=jsonData["Parameters"]["ndim"];
    // for(int i=0;i<10;i++)
    // {
    //     PDE.A[i]=0.0;
    // }

    std::cout<<"I am working on a "<<Param.ndim<<"D Problem"<<std::endl;
    if(Param.ndim>=1)
    {
    // Access JSON data
        Domain.xleft = jsonData["Zone"]["xleft"];
        Domain.xright = jsonData["Zone"]["xright"];

        // BC.RobinBack.alphaUx = jsonData["BCs"]["Back"]["Robin"]["alphaUx"];
        // BC.RobinBack.betaUx = jsonData["BCs"]["Back"]["Robin"]["betaUx"];
        // BC.Back.Ux = jsonData["BCs"]["Back"]["Robin"]["Ux"];


        // BC.RobinFront.alphaUx = jsonData["BCs"]["Front"]["Robin"]["alphaUx"];
        // BC.RobinFront.betaUx = jsonData["BCs"]["Front"]["Robin"]["betaUx"];
        // BC.Front.Ux = jsonData["BCs"]["Front"]["Robin"]["Ux"];


        // BC.RobinLeft.alphaUx = jsonData["BCs"]["Left"]["Robin"]["alphaUx"];
        // BC.RobinLeft.betaUx = jsonData["BCs"]["Left"]["Robin"]["betaUx"];
        // BC.Left.Ux = jsonData["BCs"]["Left"]["Robin"]["Ux"];

        // BC.RobinRight.alphaUx = jsonData["BCs"]["Right"]["Robin"]["alphaUx"];
        // BC.RobinRight.betaUx = jsonData["BCs"]["Right"]["Robin"]["betaUx"];
        // BC.Right.Ux = jsonData["BCs"]["Right"]["Robin"]["Ux"];
        
        
        // BC.RobinTop.alphaUx = jsonData["BCs"]["Top"]["Robin"]["alphaUx"];
        // BC.RobinTop.betaUx = jsonData["BCs"]["Top"]["Robin"]["betaUx"];
        // BC.Top.Ux = jsonData["BCs"]["Top"]["Robin"]["Ux"];

        // BC.RobinBottom.alphaUx = jsonData["BCs"]["Bottom"]["Robin"]["alphaUx"];
        // BC.RobinBottom.betaUx = jsonData["BCs"]["Bottom"]["Robin"]["betaUx"];
        // BC.Bottom.Ux = jsonData["BCs"]["Bottom"]["Robin"]["Ux"];
        
        // //For Pressure Robin BC

        // BC.RobinBack.alphaP = jsonData["BCs"]["Back"]["Robin"]["alphaP"];
        // BC.RobinBack.betaP = jsonData["BCs"]["Back"]["Robin"]["betaP"];
        // BC.Back.P = jsonData["BCs"]["Back"]["Robin"]["P"];

        // BC.RobinFront.alphaP = jsonData["BCs"]["Front"]["Robin"]["alphaP"];
        // BC.RobinFront.betaP = jsonData["BCs"]["Front"]["Robin"]["betaP"];
        // BC.Front.Ux = jsonData["BCs"]["Front"]["Robin"]["P"];


        // BC.RobinLeft.alphaP = jsonData["BCs"]["Left"]["Robin"]["alphaP"];
        // BC.RobinLeft.betaP = jsonData["BCs"]["Left"]["Robin"]["betaP"];
        // BC.Left.Ux = jsonData["BCs"]["Left"]["Robin"]["P"];

        // BC.RobinRight.alphaP = jsonData["BCs"]["Right"]["Robin"]["alphaP"];
        // BC.RobinRight.betaP = jsonData["BCs"]["Right"]["Robin"]["betaP"];
        // BC.Right.Ux = jsonData["BCs"]["Right"]["Robin"]["P"];
        
        
        // BC.RobinTop.alphaP = jsonData["BCs"]["Top"]["Robin"]["alphaP"];
        // BC.RobinTop.betaP = jsonData["BCs"]["Top"]["Robin"]["betaP"];
        // BC.Top.Ux = jsonData["BCs"]["Top"]["Robin"]["P"];

        // BC.RobinBottom.alphaP = jsonData["BCs"]["Bottom"]["Robin"]["alphaP"];
        // BC.RobinBottom.betaP = jsonData["BCs"]["Bottom"]["Robin"]["betaP"];
        // BC.Bottom.Ux = jsonData["BCs"]["Bottom"]["Robin"]["P"];

        // IC.Ux = jsonData["ICs"]["Ux"];
        Param.Nx = jsonData["Parameters"]["Nx"];
        // for(int i=0;i<3;i++)
        // {
        //     string coeff="A"+to_string(i);
        //     PDE.A[i]=jsonData["PDE"][coeff];
        // }
    }
    if(Param.ndim>=2)
    {
        Domain.ytop = jsonData["Zone"]["ytop"];
        Domain.ybottom = jsonData["Zone"]["ybottom"];
        
        // BC.RobinBack.alphaUy = jsonData["BCs"]["Back"]["Robin"]["alphaUy"];
        // BC.RobinBack.betaUy = jsonData["BCs"]["Back"]["Robin"]["betaUy"];
        // BC.Back.Uy = jsonData["BCs"]["Back"]["Robin"]["Uy"];

        // BC.RobinFront.alphaUy = jsonData["BCs"]["Front"]["Robin"]["alphaUy"];
        // BC.RobinFront.betaUy = jsonData["BCs"]["Front"]["Robin"]["betaUy"];
        // BC.Front.Uy = jsonData["BCs"]["Front"]["Robin"]["Uy"];


        // BC.RobinLeft.alphaUy = jsonData["BCs"]["Left"]["Robin"]["alphaUy"];
        // BC.RobinLeft.betaUy = jsonData["BCs"]["Left"]["Robin"]["betaUy"];
        // BC.Left.Uy = jsonData["BCs"]["Left"]["Robin"]["Uy"];

        // BC.RobinRight.alphaUy = jsonData["BCs"]["Right"]["Robin"]["alphaUy"];
        // BC.RobinRight.betaUy = jsonData["BCs"]["Right"]["Robin"]["betaUy"];
        // BC.Right.Uy = jsonData["BCs"]["Right"]["Robin"]["Uy"];
        
        
        // BC.RobinTop.alphaUy = jsonData["BCs"]["Top"]["Robin"]["alphaUy"];
        // BC.RobinTop.betaUy = jsonData["BCs"]["Top"]["Robin"]["betaUy"];
        // BC.Top.Uy = jsonData["BCs"]["Top"]["Robin"]["Uy"];

        // BC.RobinBottom.alphaUy = jsonData["BCs"]["Bottom"]["Robin"]["alphaUy"];
        // BC.RobinBottom.betaUy = jsonData["BCs"]["Bottom"]["Robin"]["betaUy"];
        // BC.Bottom.Uy = jsonData["BCs"]["Bottom"]["Robin"]["Uy"];
        
        // IC.Uy = jsonData["ICs"]["Uy"];
        Param.Ny = jsonData["Parameters"]["Ny"];
        Param.Nz=1;
        // for(int i=3;i<6;i++)
        // {
        //     string coeff="A"+to_string(i);      
        //     PDE.A[i]=jsonData["PDE"][coeff];
        // }

    }
    if(Param.ndim>=3)
    {
        std::cout<<"I am here"<<std::endl;
        Domain.zback = jsonData["Zone"]["zback"];
        Domain.zfront = jsonData["Zone"]["zfront"];
        
        // BC.RobinBack.alphaUz = jsonData["BCs"]["Back"]["Robin"]["alphaUz"];
        // BC.RobinBack.betaUz = jsonData["BCs"]["Back"]["Robin"]["betaUz"];
        // BC.Back.Uz = jsonData["BCs"]["Back"]["Robin"]["Uz"];

        // BC.RobinFront.alphaUz = jsonData["BCs"]["Front"]["Robin"]["alphaUz"];
        // BC.RobinFront.betaUz = jsonData["BCs"]["Front"]["Robin"]["betaUz"];
        // BC.Front.Uz = jsonData["BCs"]["Front"]["Robin"]["Uz"];


        // BC.RobinLeft.alphaUz = jsonData["BCs"]["Left"]["Robin"]["alphaUz"];
        // BC.RobinLeft.betaUz = jsonData["BCs"]["Left"]["Robin"]["betaUz"];
        // BC.Left.Uz = jsonData["BCs"]["Left"]["Robin"]["Uz"];

        // BC.RobinRight.alphaUz = jsonData["BCs"]["Right"]["Robin"]["alphaUz"];
        // BC.RobinRight.betaUz = jsonData["BCs"]["Right"]["Robin"]["betaUz"];
        // BC.Right.Uz = jsonData["BCs"]["Right"]["Robin"]["Uz"];
        
        
        // BC.RobinTop.alphaUz = jsonData["BCs"]["Top"]["Robin"]["alphaUz"];
        // BC.RobinTop.betaUz = jsonData["BCs"]["Top"]["Robin"]["betaUz"];
        // BC.Top.Uz = jsonData["BCs"]["Top"]["Robin"]["Uz"];

        // BC.RobinBottom.alphaUz = jsonData["BCs"]["Bottom"]["Robin"]["alphaUz"];
        // BC.RobinBottom.betaUz = jsonData["BCs"]["Bottom"]["Robin"]["betaUz"];
        // BC.Bottom.Uz = jsonData["BCs"]["Bottom"]["Robin"]["Uz"];

        // IC.Uz = jsonData["ICs"]["Uz"];
        Param.Nz = jsonData["Parameters"]["Nz"];
        // for(int i=6;i<10;i++)
        // {
        //     string coeff="A"+to_string(i);
        //     PDE.A[i]=jsonData["PDE"][coeff];
        // }
    }
    
    


    // BC.Tw = jsonData["BCs"]["Tw"];
    // BC.rho = jsonData["BCs"]["rho"];
 
    
    // IC.T = jsonData["ICs"]["T"];
    // IC.rho = jsonData["ICs"]["rho"];

    Constant.R = jsonData["Constants"]["R"];
    Constant.alpha = jsonData["Constants"]["alpha"];

    Param.d = jsonData["Parameters"]["d"];
    Param.tao = jsonData["Parameters"]["tao"];
    Param.dt = jsonData["Parameters"]["dt"];
    Param.tfinal = jsonData["Parameters"]["tfinal"];
    Param.r = jsonData["Parameters"]["r"];
       
    
    Param.radiusFactor = jsonData["Parameters"]["radiusFactor"];
    Param.saveFreq = jsonData["Parameters"]["saveFreq"];  // save results every saveFreq time steps.
}


/* Shows the calcualated parameters*/
void showparameters()
{
    std::cout << "Zone Detaols" << std::endl;
    std::cout << "xleft: " << Domain.xleft << std::endl;
    std::cout << "xright: " << Domain.xright << std::endl;
    std::cout << "ytop: " << Domain.ytop << std::endl;
    std::cout << "ybottom: " << Domain.ybottom << std::endl;
    std::cout << "zback: " << Domain.zback << std::endl;
    std::cout << "zfront: " << Domain.zfront << std::endl;

    // std::cout << "Boundary Conditions" << std::endl;
    // std::cout << "Velocity" << std::endl;
    // std::cout << "Back Side Condition alpha =" << BC.RobinBack.alphaUx << ", beta = " << BC.RobinBack.betaUx << std::endl;
    // std::cout << "Back Side (" << BC.Back.Ux << "," << BC.Back.Uy << "," << BC.Back.Uz << ")" << std::endl;
    // std::cout << "Front Side Condition alpha =" << BC.RobinFront.alphaUx << ", beta = " << BC.RobinFront.betaUx << std::endl;
    // std::cout << "Front Side (" << BC.Front.Ux << "," << BC.Front.Uy << "," << BC.Front.Uz << ")" << std::endl;
    // std::cout << "Top Side Condition alpha =" << BC.RobinTop.alphaUx << ", beta = " << BC.RobinTop.betaUx << std::endl;
    // std::cout << "Top Side (" << BC.Top.Ux << "," << BC.Top.Uy << "," << BC.Top.Uz << ")" << std::endl;
    // std::cout << "Bottom Side Condition alpha =" << BC.RobinBottom.alphaUx << ", beta = " << BC.RobinBottom.betaUx << std::endl;    
    // std::cout << "Bottom Side (" << BC.Bottom.Ux << "," << BC.Bottom.Uy << "," << BC.Bottom.Uz << ")" << std::endl;
    // std::cout << "Left Side Condition alpha =" << BC.RobinLeft.alphaUx << ", beta = " << BC.RobinLeft.betaUx << std::endl;    
    // std::cout << "Left Side (" << BC.Left.Ux << "," << BC.Left.Uy << "," << BC.Left.Uz << ")" << std::endl;
    // std::cout << "Right Side Condition alpha =" << BC.RobinRight.alphaUx << ", beta = " << BC.RobinRight.betaUx << std::endl;    
    // std::cout << "Right Side (" << BC.Right.Ux << "," << BC.Right.Uy << "," << BC.Right.Uz << ")" << std::endl;
    // std::cout << "TW " << BC.Tw << std::endl;
    // std::cout << "rho " << BC.rho << std::endl;

    // std::cout << "Initial Conditions" << std::endl;
    // std::cout << "Velelocity (" << IC.Ux << "," << IC.Uy << "," << IC.Uz << ")" << std::endl;
    // std::cout << "TW " << IC.T << std::endl;
    // std::cout << "rho " << IC.rho << std::endl;

    std::cout << "Constants" << std::endl;
    std::cout << "R " << Constant.R << std::endl;
    std::cout << "alpha " << Constant.alpha << std::endl;

    std::cout << "Parameters " << std::endl;
    std::cout << "dimension " << Param.ndim << std::endl;
    std::cout << "d " << Param.d << std::endl;
    std::cout << "tao " << Param.tao << std::endl;
    std::cout << "dt " << Param.dt << std::endl;
    std::cout << "tfinal " << Param.tfinal << std::endl;
    std::cout << "r " << Param.r << std::endl;
    std::cout << "rb " << Param.rb << std::endl;
    std::cout << "Nx " << Param.Nx << std::endl;
    std::cout << "Ny " << Param.Ny << std::endl;
    std::cout << "Nz " << Param.Nz << std::endl;

    std::cout << "dx " << CalcParam.dx << std::endl;
    std::cout << "dy " << CalcParam.dy << std::endl;
    std::cout << "dz " << CalcParam.dz << std::endl;
    std::cout << "N " << CalcParam.N << std::endl;
    std::cout << "VoxelxBox " << CalcParam.xBox << std::endl;
    std::cout << "VoxelyBox " << CalcParam.yBox << std::endl;
    std::cout << "VoxelzBox " << CalcParam.zBox << std::endl;
    std::cout << "radius " << CalcParam.radius << std::endl;
    std::cout << "Number of Voxel Boxes X Direction " << CalcParam.nbxBox << std::endl;
    std::cout << "Number of Voxel Boxes Y Direction " << CalcParam.nbyBox << std::endl;
    std::cout << "Number of Voxel Boxes Z Direction " << CalcParam.nbzBox << std::endl;
    std::cout << "Total Number of Voxel Boxes " << CalcParam.nbxBox * CalcParam.nbyBox * CalcParam.nbyBox<< std::endl;
    // std::cout << "PDE Coefficients"<<std::endl;
    // for(int i=0;i<10;i++)
    // {
    //     std::cout<<PDE.A[i]<<"\t";
    // }
    // std::cout<<std::endl;
    // int panch;
    // std::cin>>panch;
}
#endif