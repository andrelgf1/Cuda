#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * An example of using a statically declared global variable (devData) to store
 * a floating-point value on the device.
 */

__device__ float devData[5];

__global__ void checkGlobalVariable(int arr_size)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // display the original value
    printf("Device: Initial value of global variable  %f\n", devData[thread_id]);
    if(thread_id < arr_size) devData[thread_id] *= thread_id;
}

int main(void)
{
    // initialize the global variable
    float value[5] = {3.14f, 3.14f, 3.14f, 3.14f, 3.14f};
    printf("Host: Initial array values  [ %f %f %f %f %f ]\n",value[0], value[1], value[2], value[3], value[4]);
    CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(value)));

    int arraySize = sizeof(value) / sizeof(value[0]);
    // invoke the kernel
    checkGlobalVariable<<<1, arraySize>>>(arraySize);

    // copy the global variable back to the host
    CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(value)));
    printf("Host: New array values  [ %f %f %f %f %f ]\n",value[0], value[1], value[2], value[3], value[4]);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
