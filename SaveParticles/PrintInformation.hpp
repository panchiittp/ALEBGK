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


void printperiodicneigh(BGKParticle *dP, CalcParameters CalcParam, int partind, std::string filename)
{
    std::ofstream file;//(filename);
    file.open(filename,std::ios::app);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    std::string direction[]={"Left","Right","Bottom","Top","Front","Back"};

    int maxneigh = 0;
    for (int l = 0; l < CalcParam.N; l++)
    {

        if (maxneigh < dP[l].totneigh)
        {
            maxneigh = dP[l].totneigh;
        }

        if (l == partind)
        {
            file << l << ", " << (dP[l].boundary? "Boundary": "Interior") << "Neigbhour Radius "<<CalcParam.radius
                        << " (" << dP[l].x << " ," << dP[l].y << ", " << dP[l].z << ")"
                        << " My voxel: " << dP[l].voxel << " My Status : " << std::endl;
            file << " Number of Neighbour Voxel: " << dP[l].totvoxel << std::endl;
            for (int j = 0; j < dP[l].totvoxel; j++)
            {
                file << dP[l].neighVoxel[j] << " ";
            }
            file << std::endl;
            

            file << " Number of Neigbours " << dP[l].totneigh << std::endl;
            int totneigh[6]={0,0,0,0,0,0};
            for (int j = 0; j < dP[l].totneigh; j++)
            {
                file << "Neighbour Number = "<<j<<std::endl<< "Neighbour Index " <<dP[l].neighindex[j] << std::endl<< "From voxel:  " << dP[dP[l].neighindex[j]].voxel <<std::endl
                            << "Point = (" << dP[dP[l].neighindex[j]].x<<","<<dP[dP[l].neighindex[j]].y<<","<<dP[dP[l].neighindex[j]].z<<")"
                            <<"Distance = " << sqrt((dP[l].x-dP[dP[l].neighindex[j]].x)*(dP[l].x-dP[dP[l].neighindex[j]].x)+
                                                (dP[l].y-dP[dP[l].neighindex[j]].y)*(dP[l].y-dP[dP[l].neighindex[j]].y)+
                                                (dP[l].z-dP[dP[l].neighindex[j]].z)*(dP[l].z-dP[dP[l].neighindex[j]].z))<< std::endl<< "Type: "<<std::endl;                    
                for(int k=3*j;k<3*(j+1);k++)
                {
                    file<<"k="<<k<<" dP[l].neightype[k] = "<<dP[l].neightype[k]<<" ";
                    // int panch;
                    // std::cin>>panch;
                    // continue;
                    if(dP[l].neightype[k]!=-1)
                    {

                        switch(dP[l].neightype[k])
                        {
                            case 0:
                                file<<"Left";
                                totneigh[0]++;
                                break;
                            case 1:
                                file<<"Right";
                                totneigh[1]++;
                                break;
                            case 2:
                                file<<"Bottom";
                                totneigh[2]++;
                                break;
                            case 3:
                                file<<"Top";
                                totneigh[3]++;
                                break;
                            case 4:
                                file<<"Front";
                                totneigh[4]++;
                                break;
                            case 5:
                                file<<"Back";
                                totneigh[5]++;
                                break;
                            default:
                                file<<"Illegal Type "<<dP[l].neightype[k] <<std::endl;

                        }
                        file<<std::endl;
                    }
                    else
                        file<<std::endl;
                }
                
                // file<<"M Matrix"<<std::endl;
                // for(int k=0;k<10;k++)
                // {
                //     file<<dP[l].M[j*10+k]<<" ";
                // }              
                // file<<std::endl<<"W Matrix"<<std::endl;
                // file<<dP[l].W[j];
                // file << std::endl;
            }
            for(int i=0;i<6;i++)
            {
                std::cout<<"Total Number of Neighbours in "<<direction[i]<< " is "<<totneigh[i]<<std::endl;
                std::cout<<"Counted Using GPU "<<dP[l].neighcount[i]<<std::endl;
            }
            std::cout<<"Total Number of Neighbours is "<<dP[l].totneigh<<std::endl;
            file << std::endl;
            //break;
            file << "MTW Matrix"<<std::endl;
            for(int i=0;i<6;i++)
            {
                for(int j=0;j<dP[l].totneigh;j++)
                {
                    file<<dP[l].MTW[i*dP[l].totneigh+j]<<" ";
                }
                file<<std::endl;
            }
            
            file << "MTWM Inv Matrix"<<std::endl;
            for(int i=0;i<6;i++)
            {
                for(int j=0;j<6;j++)
                {
                    if(j>=i)
                    {
                        file<<dP[l].MTWM[6*i+j-i*(i+1)/2]<<" ";
                    }
                    else
                    {
                        file<<dP[l].MTWM[6*j+i-j*(j+1)/2]<<" ";
                    }
                }
                file<<std::endl;
            }

            // file << "MTW Final Matrix"<<std::endl;
            // for(int i=0;i<10;i++)
            // {
            //     for(int j=0;j<dP[l].totneigh;j++)
            //     {
            //         file<<dP[l].MTWFinal[i*dP[l].totneigh+j]<<" ";
            //     }
            //     file<<std::endl;
            // }
            file << "RHS"<<std::endl;
            for(int i=0;i<dP[l].totneigh;i++)
            {
                file<<std::scientific <<dP[l].rhs[i];
                file<<std::endl;
            }
            file << "gWENO"<<std::endl;
            for(int i=0;i<70;i++)
            {
                file<<std::scientific << dP[l].gWENO[i];
                file<<std::endl;
            }                
            // for (int l1 = 0; l1 < 15; l1++)
            // {
            //     for (int k = 0; k < 15; k++)
            //     {
            //         for (int j = 0; j < 15; j++)
            //         {
            //             int linearIndex = j + 15 * k + l1 * 15 * 15;
            //             for (int i1 = 0; i1 < dP[l].totneigh; i1++)
            //             {                    
            //                 file<<"("<<dP[dP[l].neighindex[i1]].g[linearIndex]<<","<< dP[l].g[linearIndex]<<") ";                    
            //             }
            //             file<<std::endl;
            //         }
            //     }
            // }
        }

    }

    file << "Maximum Number of Neighbours " << maxneigh << std::endl;
    file << "My CheckVariable "<<dP[partind].checkvariable<<" and from GPU "<<dP[partind].neighcount[0]<< std::endl;
    for(int i=0;i<6;i++)
    {
        file<<"Total Number of Neighbours in "<<direction[i]<< "Counted Using GPU is "<<dP[partind].neighcount[i]<<std::endl;
    }
    file.close();
}
#endif