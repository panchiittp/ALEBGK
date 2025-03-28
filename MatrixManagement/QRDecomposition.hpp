#ifndef QRDECOMPOSITION_HPP
#define QRDECOMPOSITION_HPP
__device__ __host__ double dotProduct(double* a, double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Scalar multiplication of a vector
__device__ __host__ void scalarMultiply(double* a, double scalar, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] *= scalar;
    }
}

// Vector subtraction: a = a - b
__device__ __host__ void vectorSubtract(double* a, double* b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] -= b[i];
    }
}

// Copy vector b to a
__device__ __host__ void copyVector(double* a, double* b, int n) {
    for (int i = 0; i < n; ++i) {
        a[i] = b[i];
    }
}

// QR Decomposition using Gram-Schmidt orthogonalization
__device__ __host__ void QRDecomposition(double* A, double* Q, double* R, int m, int n) {
    for (int j = 0; j < n; ++j) {
        // Copy the j-th column of A to the j-th column of Q
        for (int i = 0; i < m; ++i) {
            Q[i * n + j] = A[i * n + j];
        }

        // Orthogonalization
        for (int k = 0; k < j; ++k) {
            double r = dotProduct(&Q[k * m], &Q[j * m], m);
            R[k * n + j] = r;

            // Subtract the projection
            double *proj=(double *)malloc(m * sizeof(double));

            copyVector(proj, &Q[k * m], m);
            scalarMultiply(proj, r, m);
            vectorSubtract(&Q[j * m], proj, m);
            free(proj);
        }

        // Normalization
        double norm = sqrt(dotProduct(&Q[j * m], &Q[j * m], m));
        R[j * n + j] = norm;

        if (norm > 1e-12) {
            scalarMultiply(&Q[j * m], 1.0 / norm, m);
        }
    }
}


__device__ __host__ void backwardSubstitution(double* U, double* b, double* x, int n) {
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= U[i * n + j] * x[j];
        }
        x[i] /= U[i * n + i];
    }
}

__device__ __host__ void InverseQR(double *A, double *Ainv,int n)
{
    double *Q=(double *)malloc(n*n * sizeof(double));
    double *R=(double *)malloc(n*n * sizeof(double));
    double *b=(double *)malloc(n * sizeof(double));
    double *x=(double *)malloc(n * sizeof(double));
    QRDecomposition(A, Q, R,n,n);
    for (int j = 0; j < n; ++j) 
    {
        // Set up the identity column as the RHS
        for (int i = 0; i < n; ++i) {
            b[i] = (i == j) ? 1.0 : 0.0;
        }

        // Solve R * x = Q^T * b using backward substitution
        double* Qt_b = (double *)malloc(n * sizeof(double));
        for (int i = 0; i < n; ++i) {
            Qt_b[i] = 0.0;
            for (int k = 0; k < n; ++k) {
                Qt_b[i] += Q[k * n + i] * b[k];
            }
        }

        backwardSubstitution(R, Qt_b, x, n);

        // Store the solution vector in the inverse matrix
        for (int i = 0; i < n; ++i) {
            Ainv[i * n + j] = x[i];
        }
        free(Qt_b);
    }
    free(Q);
    free(R);
    free(x);
    free(b);
}
#endif