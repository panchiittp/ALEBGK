#ifndef SAVEPARTICLES_H
#define SAVEPARTICLES_H
void writeVTKFile(const std::string &filename, BGKParticle *particles, int numParticles)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    // Write the VTK file header
    file << "# vtk DataFile Version 3.0" << std::endl;
    file << "Particle data" << std::endl;
    file << "ASCII" << std::endl;
    file << "DATASET POLYDATA" << std::endl;

    // Write the points (particle positions)
    file << "POINTS " << numParticles << " float" << std::endl;
    for (int i = 0; i < numParticles; ++i)
    {
        file << particles[i].x << " " << particles[i].y << " " << particles[i].z << std::endl;
    }

    // Write the point data (e.g., active status)
    file << "POINT_DATA " << numParticles << std::endl;
    file << "SCALARS velocity float" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    // for (int i = 0; i < numParticles; ++i)
    // {
    //     file << particles[i].prob.rho << std::endl;
    // }
    file << "SCALARS analytic float" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < numParticles; ++i)
    {
        file << particles[i].ux<< std::endl;
    }
    // file << "VECTORS velocity float" << std::endl;
    // for (int i = 0; i < numParticles; ++i)
    // {
    //     file << particles[i].ux << " " << particles[i].uy << " " << particles[i].uz << std::endl;
    // }
    file << "SCALARS boundary int 1" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    for (int i = 0; i < numParticles; ++i)
    {
        file << particles[i].boundary << std::endl;
    }
    file.close();
    std::cout << "VTK file written: " << filename << std::endl;
}
void SaveParticleForMatlab(const std::string &filename, BGKParticle *particles, int numParticles)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    //file << "POINTS " << numParticles << " float" << std::endl;
    for (int i = 0; i < numParticles; ++i)
    {
        file << particles[i].x << " " << particles[i].y << " " << particles[i].z << " " << particles[i].ux  << " " << particles[i].uy << std::endl;
    }
    file.close();
    std::cout << "Particle Saved for MATLAB: " << filename << std::endl;

}

void SaveNeighbourParticleForMatlab(const std::string &filename, BGKParticle *particles, int numParticles,int ind)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    //file << "POINTS " << numParticles << " float" << std::endl;
    for (int i = 0; i < particles[ind].totneigh; ++i)
    {
        int k=particles[ind].neighindex[i];
        file << particles[k].x << " " << particles[k].y << " " << particles[k].z << std::endl;
    }
    file.close();
    std::cout << "Neighbour Particle Saved for MATLAB: " << filename << std::endl;

}

void SavePeriodicNeighbourParticleForMatlab(const std::string &filename, BGKParticle *particles, int numParticles,int ind,int neightype)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    //file << "POINTS " << numParticles << " float" << std::endl;
    for (int i = 0; i < particles[ind].totneigh; ++i)
    {
        int k=particles[ind].neighindex[i];
        for(int s=3*i;s<3*(i+1);s++)
            if(particles[ind].neightype[s]==neightype)
                file << particles[k].x << " " << particles[k].y << " " << particles[k].z << std::endl;
    }

    
    file.close();
    std::cout << "Periodic Neighbour Particle Saved for MATLAB: " << filename << std::endl;

}
#include "WriteToCSV.hpp"
#endif

