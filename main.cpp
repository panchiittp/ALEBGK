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
#include <iomanip>
#include "Definitions/Definitions2D.hpp"
#include "Parameters/Parameters.hpp"
#include "SaveParticles/SaveParticles.hpp"
#include "ParticleManagement/ParticleManagement.hpp"
#include "MemoryManagement/MemoryManagement.hpp"
#include "InitialConditions/InitialConditions.hpp"
#include "Interpolation/Interpolation.hpp"
#include "Solvers/GPUSolvers.hpp"
#include "Solvers/CPUSolvers.hpp"
using namespace std;


int main()
{
    auto startprog = std::chrono::high_resolution_clock::now();
    readparameters();
    CalculateParameters();
    showparameters();

    if(ExecuteOn=="GPU")
    {
        GPUSolvers();
    }
    else
    {
        CPUSolvers();
    }
    
    return 0;
}