#ifndef PRINTINFORMATION_HPP
#define PRINTINFORMATION_HPP

void printsizerequired(size_t value)
{

    if (value < 1024)
    {
        std::cout << "Memory Required for allocation is " << value << " bytes" << std::endl;
    }
    else if (value >= 1024 && value < 1024 * 1024)
    {
        std::cout << "Memory Required for allocation is " << value / 1024 << " KB" << std::endl;
    }
    else if (value >= 1024 * 1024 && value < 1024 * 1024 * 1024)
    {
        std::cout << "Memory Required for allocation is " << value / (1024 * 1024) << " MB" << std::endl;
    }
    else
    {
        std::cout << "Memory Required for allocation is " << value / (1024 * 1024 * 1024) << " GB" << std::endl;
    }
}

void printvoxelinfo(voxelDetails *dvoxinfo, int i)
{
    std::cout << "voxel Number:" << i << std::endl;
    std::cout << "Number of Particles " << dvoxinfo[i].count << std::endl;
    std::cout << "List of Particles" << std::endl;
    for (int j = 0; j < dvoxinfo[i].count; j++)
    {
        std::cout << dvoxinfo[i].particleindex[j] << " ";
    }
    std::cout << std::endl;
}

void printneighanddist(int ind, BGKParticle *dP)
{
    std::cout << "Point: " << ind << "(" << dP[ind].x << "," << dP[ind].y << "," << dP[ind].z << ")" << std::endl;
    std::cout << "List of Neighbours" << std::endl;
    std::cout << "Number of Neighbours :" <<  dP[ind].totneigh << std::endl;
    for (int i = 0; i < dP[ind].totneigh; i++)
    {
        int s = dP[ind].neighindex[i];
        std::cout << "Neighbour " << i << " Point: " << s << "(" << dP[s].x << "," << dP[s].y << "," << dP[s].z << ") ";
        std::cout << "Distance: (dx,dy,dz) = (" << dP[s].x - dP[ind].x << "," << dP[s].y - dP[ind].y << "," << dP[s].z - dP[ind].z << ")" << std::endl;
    }
}

void printneighvoxel(int ind, BGKParticle *dP)
{
    std::cout << "Point: " << ind << "(" << dP[ind].x << "," << dP[ind].y << "," << dP[ind].z << ")" << std::endl;
    std::cout << "List of Voxels" << std::endl;
    std::cout << "Number of Neighbour Voxels :" <<  dP[ind].totvoxel << std::endl;
    for (int i = 0; i < dP[ind].totvoxel; i++)
    {
        std::cout<<dP[ind].neighVoxel[i]<<" ";
    }
    std::cout<<std::endl;
}
#endif