#include "../common/common.h"
#include <stdio.h>

/*
 * A simple introduction to programming in CUDA. This program prints "Hello
 * World from GPU! from 10 CUDA threads running on the GPU.
 */

__global__ void helloFromGPU()
{
   if(threadIdx.x == 5)  printf("Hello World from GPU thread %d!\n",threadIdx.x);
}

int main(int argc, char **argv)
{
    const char *msg = "Hello World from CPU!\n";
    printf("%s\n",msg);

    helloFromGPU<<<1, 10>>>();
    // CHECK(cudaDeviceReset());
    CHECK(cudaDeviceSynchronize());

    return 0;
}


