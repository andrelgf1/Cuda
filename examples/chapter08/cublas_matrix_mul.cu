#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"

/*
 * A simple example of performing matrix-vector multiplication using the cuBLAS
 * library and some randomly generated inputs.
 */

/*
 * M = # of rows
 * N = # of columns
 */
int M = 10;
int N = 10;

/*
 * Generate a vector of length N with random single-precision floating-point
 * values between 0 and 100.
 */
void generate_random_vector(int N, float **outX)
{
    int i;
    double rMax = (double)RAND_MAX;
    float *X = (float *)malloc(sizeof(float) * N);

    for (i = 0; i < N; i++)
    {
        int r = rand();
        double dr = (double)r;
        X[i] = (dr / rMax) * 100.0;
    }

    *outX = X;
}

/*
 * Generate a matrix with M rows and N columns in column-major order. The matrix
 * will be filled with random single-precision floating-point values between 0
 * and 100.
 */
void generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);

    // For each column
    for (j = 0; j < N; j++)
    {
        // For each row
        for (i = 0; i < M; i++)
        {
            double dr = (double)rand();
            A[j * M + i] = (dr / rMax) * 100.0;
        }
    }

    *outA = A;
}

int main(int argc, char **argv)
{
    // int i;
    float *A, *dA;
    float *A2, *dA2;
    float *R, *dR;
    float beta;
    float alpha;
    cublasHandle_t handle = 0;

    alpha = 3.0f;
    beta = 4.0f;

    R = (float *)malloc(sizeof(float)*M*M);
    memset(R, 0x00, sizeof(float)*M*M);

    // Generate inputs
    srand(9384);
    generate_random_dense_matrix(M, N, &A);
    generate_random_dense_matrix(N, M, &A2);
   
    // Create the cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&handle));

    // Allocate device memory
    CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CHECK(cudaMalloc((void **)&dA2, sizeof(float) * N * M));
    CHECK(cudaMalloc((void **)&dR, sizeof(float) * M * M));

    // Transfer inputs to the device
    CHECK_CUBLAS(cublasSetMatrix(M, N, sizeof(float), A, M, dA, M));
    CHECK_CUBLAS(cublasSetMatrix(N, M, sizeof(float), A2, N, dA2, N));
    CHECK_CUBLAS(cublasSetMatrix(M, M, sizeof(float), R, M, dR, M));

    // Execute the matrix-matrix multiplication
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, M, N, &alpha,
        dA, M, dA2, N, &beta, dR, M));

    // Retrieve the output vector from the device
    CHECK_CUBLAS(cublasGetMatrix(M, M, sizeof(float), dR, M, R, M));

    printf("\n");
    for (int i = 0; i < M; i++)
    {
        printf("|");
        for (int j = 0; j < M; j++)        
            printf("%8.4f ", R[i*M + j]);
        printf("|\n|\n");
    }


    free(A);
    free(A2);
    free(R);

    CHECK(cudaFree(dA));
    CHECK(cudaFree(dA2));
    CHECK(cudaFree(dR));
    CHECK_CUBLAS(cublasDestroy(handle));

    return 0;
}
