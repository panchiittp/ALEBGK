#ifndef INTERPOLATIONCPU_HPP
#define INTERPOLATIONCPU_HPP

void CenterWENOCPU(int p, BGKParticle *dP, Parameters Param, CalcParameters CalcParam, Constants Constant,DomainBoundary Domain,int flag)
{
    int row=dP[p].totneigh;
    double Lx=(Domain.xright - Domain.xleft);
    double Ly=(Domain.ytop - Domain.ybottom);
    // double Lz=(Domain.zback - Domain.zfront);
    //printf("I am Inside this function\n");
    
    for (int i = 0; i < dP[p].totneigh; i++)
    {
        int neigh = dP[p].neighindex[i];
        double dx = (dP[neigh].x - dP[p].x);
        double dy = (dP[neigh].y - dP[p].y);
        // double dz = (dP[neigh].z - dP[p].z);
        double dist = std::sqrt(dx * dx + dy * dy);// + dz * dz);

        if(dist > 3*CalcParam.radius)
        {
            if (dx >=  Lx / 2.0) dx -= Lx;
            if (dx <= -Lx / 2.0) dx += Lx;

            if (dy >=  Ly / 2.0) dy -= Ly;
            if (dy <= -Ly / 2.0) dy += Ly;

            // if (dz >=  Lz / 2.0) dz -= Lz;
            // if (dz <= -Lz / 2.0) dz += Lz;
        }
        //double dummy = std::pow(dx, 2) + std::pow(dy, 2);// + std::pow(dz, 2);
        dist = sqrt(dx * dx + dy * dy);
        double weight=std::exp(-Constant.alpha * (dist)/(CalcParam.radius*CalcParam.radius));
        dP[p].M[i*row]=1;
        dP[p].M[i*row+1]=dx;
        dP[p].M[i*row+2]=0.5*dx*dx;
        dP[p].M[i*row+3]=dy;
        dP[p].M[i*row+4]=dx*dy;
        dP[p].M[i*row+5]=0.5*dy*dy;
        dP[p].W[i]=weight;
        if((i==0 || i==1) && p==1251)
        {
            std::cout<<std::scientific<<"i= "<<i<<"weight = "<<weight<<"dp weigh"<<dP[p].W[i]<<std::endl;
        }
//            printf("i= %d weight = %lf\n",i,weight);

        dP[p].MTW[i]=1.0*weight;
        dP[p].MTW[row+i]=dx*weight;
        dP[p].MTW[2*row+i]=0.5*dx*dx*weight;
        dP[p].MTW[3*row+i]=dy*weight;
        dP[p].MTW[4*row+i]=dx*dy*weight;
        dP[p].MTW[5*row+i]=0.5*dy*dy*weight;
        // dP[p].MTW[6*row+i]=dz*weight;
        // dP[p].MTW[7*row+i]=dx*dz*weight;
        // dP[p].MTW[8*row+i]=dy*dz*weight;
        // dP[p].MTW[9*row+i]=0.5*dz*dz*weight;



        dP[p].MTWM[0]+=weight; //(0,0)
        dP[p].MTWM[1]+=weight*dx; //(0,1)
        dP[p].MTWM[2]+=weight*dx*dx*0.5; //(0,2)
        dP[p].MTWM[3]+=weight*dy; //(0,3)
        dP[p].MTWM[4]+=weight*dx*dy; //(0,4)
        dP[p].MTWM[5]+=weight*dy*dy*0.5; //(0,5)
        // dP[p].MTWM[6]+=weight*dz;//(0,6)
        // dP[p].MTWM[7]+=weight*dx*dz;//(0,7)
        // dP[p].MTWM[8]+=weight*dy*dz;
        // dP[p].MTWM[9]+=weight*dz*dz*0.5;

        dP[p].MTWM[6]+=weight*dx*dx;
        dP[p].MTWM[7]+=weight*dx*dx*dx*0.5;
        dP[p].MTWM[8]+=weight*dy*dx;
        dP[p].MTWM[9]+=weight*dx*dy*dx;
        dP[p].MTWM[10]+=weight*dy*dy*dx*0.5;
        // dP[p].MTWM[15]+=weight*dz*dx;
        // dP[p].MTWM[16]+=weight*dx*dz*dx;
        // dP[p].MTWM[17]+=weight*dy*dz*dx;
        // dP[p].MTWM[18]+=weight*dz*dz*dx*0.5;

        dP[p].MTWM[11]+=weight*dx*dx*0.5*dx*dx*0.5;
        dP[p].MTWM[12]+=weight*dy*dx*dx*0.5;
        dP[p].MTWM[13]+=weight*dx*dy*dx*dx*0.5;
        dP[p].MTWM[14]+=weight*dy*dy*0.5*dx*dx*0.5;
        // dP[p].MTWM[15]+=weight*dz*dx*dx*0.5;
        // dP[p].MTWM[24]+=weight*dx*dz*dx*dx*0.5;
        // dP[p].MTWM[25]+=weight*dy*dz*dx*dx*0.5;
        // dP[p].MTWM[26]+=weight*dz*dz*0.5*dx*dx*0.5;


        dP[p].MTWM[15]+=weight*dy*dy;
        dP[p].MTWM[16]+=weight*dx*dy*dy;
        dP[p].MTWM[17]+=weight*dy*dy*dy*0.5;
        // dP[p].MTWM[30]+=weight*dz*dy;
        // dP[p].MTWM[31]+=weight*dx*dz*dy;
        // dP[p].MTWM[32]+=weight*dy*dz*dy;
        // dP[p].MTWM[33]+=weight*dz*dz*dy*0.5;

        dP[p].MTWM[18]+=weight*dx*dy*dy*dx;
        dP[p].MTWM[19]+=weight*dy*dy*dy*dx*0.5;
        // dP[p].MTWM[21]+=weight*dz*dy*dx;
        // dP[p].MTWM[37]+=weight*dx*dz*dy*dx;
        // dP[p].MTWM[38]+=weight*dy*dz*dy*dx;
        // dP[p].MTWM[39]+=weight*dz*dz*dy*0.5*dx;


        dP[p].MTWM[20]+=weight*dy*dy*dy*dy*0.5*0.5;
        // dP[p].MTWM[23]+=weight*dz*dy*dy*0.5;
        // dP[p].MTWM[42]+=weight*dx*dz*dy*dy*0.5;
        // dP[p].MTWM[43]+=weight*dy*dz*dy*dy*0.5;
        // dP[p].MTWM[44]+=weight*dz*dz*dy*dy*0.5*0.5;

        // dP[p].MTWM[45]+=weight*dz*dz;
        // dP[p].MTWM[46]+=weight*dx*dz*dz;
        // dP[p].MTWM[47]+=weight*dy*dz*dz;
        // dP[p].MTWM[48]+=weight*dz*dz*dz*0.5;

        // dP[p].MTWM[49]+=weight*dx*dz*dx*dz;
        // dP[p].MTWM[50]+=weight*dy*dz*dx*dz;
        // dP[p].MTWM[51]+=weight*dz*dz*dx*dz*0.5;

        // dP[p].MTWM[52]+=weight*dy*dz*dy*dz;
        // dP[p].MTWM[53]+=weight*dz*dz*dy*dz*0.5;

        // dP[p].MTWM[54]+=weight*dz*dz*dz*dz*0.5*0.5;
    }
}

void ConstructCenterMMatrixCPU(BGKParticle *dP, Parameters Param, CalcParameters CalcParam, Constants Constant,DomainBoundary Domain,int flag)
{
    CenterWENOCPU(1251, dP, Param, CalcParam, Constant,Domain,flag);
    // for(int p=0;p<CalcParam.N;p++)
    // {
    //     if (dP[p].boundary!=true)
    //     {
    //             CenterWENOCPU(p, dP, Param, CalcParam, Constant,Domain,flag);

    //     }
    // }
}
#endif