#ifndef SORMETHOD_HPP
#define SORMETHOD_HPP

__device__ __host__ void SORMethod(double *A, double *b, double *x, int n)
{
    double *x_old=(double *)malloc(n * sizeof(double));
    int MAX_ITER=1000;
    int iter = 0;
    double omega = 1.25;
    double TOL=1e-12;
    while (iter < MAX_ITER) {
        for (int i = 0; i < n; ++i) {
            x_old[i] = x[i];
        }

        for (int i = 0; i < n; ++i) {
            double sigma = 0.0;

            // Summation for lower triangular part
            for (int j = 0; j < i; ++j) {
                sigma += A[i * n + j] * x[j];
            }

            // Summation for upper triangular part
            for (int j = i + 1; j < n; ++j) {
                sigma += A[i * n + j] * x_old[j];
            }

            x[i] = (1 - omega) * x_old[i] + (omega / A[i * n + i]) * (b[i] - sigma);
        }

        // Check for convergence
        double error = 0.0;
        for (int i = 0; i < n; ++i) {
            error += fabs(x[i] - x_old[i]);
        }

        if (error < TOL) {            
            // std::cout<<"Iterative Method Converged after "<<iter<<" Iterations"<<std::endl;
            //free(x_old);
            break;
        }

        iter++;
    }
    // std::cout<<"Iterative Method Did not Converge after "<<iter<<" Iterations"<<std::endl;
    free(x_old);
}


__device__ __host__ void InverseSOR(double *A, double *Ainv, int n)
{
    double *b=(double *)malloc(n * sizeof(double));   
    double *x=(double *)malloc(n * sizeof(double));
    for (int j = 0; j < n; ++j) 
    {
        // Set up the identity column as the RHS
        for (int i = 0; i < n; ++i) {
            b[i] = (i == j) ? 1.0 : 0.0;
            x[i]=0.0;
        }
        
        SORMethod(A,b,x,n);
        for (int i = 0; i < n; ++i) {
            Ainv[i * n + j] = x[i];
        }        
    }
    free(x);
    // for(int i=0;i<n;i++)
    // {
    //     for(int j=0;j<n;j++)
    //     {
    //         std::cout<<std::scientific<<Ainv[i*n+j]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    
}
#endif