#ifndef CHOLESKY_HPP
#define CHOLESKY_HPP
__device__ void CholeskyDecomposition(double* A, double* L, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0.0;

            // Summation for diagonal elements
            if (i == j) {
                for (int k = 0; k < j; k++) {
                    sum += L[j * n + k] * L[j * n + k];
                }
                double value = A[j * n + j] - sum;
                if (value <= 0.0) {
                   return ;
                }
                L[j * n + j] = sqrt(value);
            } 
            // Summation for non-diagonal elements
            else {
                for (int k = 0; k < j; k++) {
                    sum += L[i * n + k] * L[j * n + k];
                }
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
}
#endif