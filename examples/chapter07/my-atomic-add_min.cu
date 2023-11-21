#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * This example illustrates implementation of custom atomic operations using
 * CUDA's built-in atomicCAS function to implement atomic signed 32-bit integer
 * addition.
 **/

__device__ int myAtomicAdd(int *address, int incr)
{
    // Create an initial guess for the value stored at *address.
    int guess = *address;
    int oldValue = atomicCAS(address, guess, guess + incr);

    // Loop while the guess is incorrect.
    while (oldValue != guess)
    {
        guess = oldValue;
        oldValue = atomicCAS(address, guess, guess + incr);
    }

    return oldValue;
}

__device__ int myAtomicMin(int *address, int incr)
{
    // Create an initial guess for the value stored at *address.
    int guess = *address;
    int new_value = blockIdx.x * blockDim.x + threadIdx.x + incr ;
    int min_value = (guess < new_value) ? guess : new_value;
    int oldValue = atomicCAS(address, guess, min_value);

    // Loop while the guess is incorrect.
    while (oldValue != guess)
    {
        guess = oldValue;
        min_value = (guess < new_value) ? guess : new_value;
        oldValue = atomicCAS(address, guess, min_value);
    }

    //  printf("New value %d --- guess %d  --- Min value %d --- Address %d\n", new_value, guess, min_value, *address);

    return oldValue;
}

__global__ void kernel(int *sharedInteger)
{
    // myAtomicAdd(sharedInteger, 1);
     myAtomicMin(sharedInteger, 1);
}

int main(int argc, char **argv)
{
    int h_sharedInteger;
    int *d_sharedInteger;
    CHECK(cudaMalloc((void **)&d_sharedInteger, sizeof(int)));
    // CHECK(cudaMemset(d_sharedInteger, 0x00, sizeof(int)));
    int hostValue = 100;
    CHECK(cudaMemcpy(d_sharedInteger, &hostValue, sizeof(int), cudaMemcpyHostToDevice));

    // kernel<<<4, 128>>>(d_sharedInteger);
     kernel<<<4, 128>>>(d_sharedInteger);

    CHECK(cudaMemcpy(&h_sharedInteger, d_sharedInteger, sizeof(int),
                     cudaMemcpyDeviceToHost));
    printf("4 x 128 numbers checked and minimun value is %d\n", h_sharedInteger);

    return 0;
}

