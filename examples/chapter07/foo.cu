__global__ void intrinsic(float *ptr) {
    *ptr = __powf(*ptr, 2.0f);
}
__global__ void standard(float *ptr) {
    *ptr = powf(*ptr, 2.0f);
}



__global__ void foo(float *ptr) {
    *ptr = (*ptr) * (*ptr) + (*ptr);
}


int main(int argc, char **argv)
{
}