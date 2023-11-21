#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>

/**
 * This example illustrates implementation of custom atomic operations using
 * CUDA's built-in atomicCAS function to implement atomic signed 32-bit integer
 * addition.
 **/

__device__ float myAtomicAddFloat(float *address, float incr) {

    // Convert address to point to a supported type of the same size
    unsigned int *typedAddress = (unsigned int *)address;
    // Stored the expected and desired float values as an unsigned int
    float currentVal = *address;
    unsigned int expected = __float2uint_rn(currentVal);
    unsigned int desired = __float2uint_rn(currentVal + incr);
    int oldIntValue = atomicCAS(typedAddress, expected, desired);

    while (oldIntValue != expected) {
        expected = oldIntValue;
        /*
        * Convert the value read from typedAddress to a float, increment,
        * and then convert back to an unsigned int
        */
        desired = __float2uint_rn(__uint2float_rn(oldIntValue) + incr);
        oldIntValue = atomicCAS(typedAddress, expected, desired);
    }

    return __uint2float_rn(oldIntValue);
}


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

// __global__ void kernel(int *sharedInteger)
__global__ void kernel(float *sharedFloat)
{
    // myAtomicAdd(sharedInteger, 1);
    myAtomicAddFloat(sharedFloat, 1.0f);
}

int main(int argc, char **argv)
{
    // int h_sharedInteger;
    // int *d_sharedInteger;
    float h_sharedFloat;
    float *d_sharedFloat;

    CHECK(cudaMalloc((void **)&d_sharedFloat, sizeof(float)));
    CHECK(cudaMemset(d_sharedFloat, 0x00, sizeof(float)));

    kernel<<<4, 128>>>(d_sharedFloat);

    CHECK(cudaMemcpy(&h_sharedFloat, d_sharedFloat, sizeof(float),
                     cudaMemcpyDeviceToHost));
    // printf("4 x 128 increments led to value of %f\n", h_sharedFloat);
    printf("4 x 128 increments led to value of %u\n", *(unsigned int*) &h_sharedFloat);

    return 0;
}

