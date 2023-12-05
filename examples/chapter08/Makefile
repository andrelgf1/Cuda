CU_APPS=cublas cuda-openacc cufft-multi cufft cusparse rand-kernel \
        replace-rand-streams replace-rand cublas_colu cublas_matrix_mul
C_APPS=simple-data simple-kernels simple-parallel

all: ${C_APPS} ${CU_APPS}
cublas_matrix_mul: cublas_matrix_mul.cu
#	nvcc -O2 -arch=sm_20 -lcublas -o cublas cublas.cu
	nvcc -g -G -lcublas -o cublas_matrix_mul cublas_matrix_mul.cu
cublas_colu: cublas_colu.cu
#	nvcc -O2 -arch=sm_20 -lcublas -o cublas cublas.cu
	nvcc -g -G -lcublas -o cublas_colu cublas_colu.cu
cublas: cublas.cu
#	nvcc -O2 -arch=sm_20 -lcublas -o cublas cublas.cu
	nvcc -g -G -lcublas -o cublas cublas.cu
cuda-openacc: cuda-openacc.cu
#	nvcc -O2 -arch=sm_20 -lcublas -lcurand -o cuda-openacc cuda-openacc.cu
	nvcc -g -G -lcublas -lcurand -o cuda-openacc cuda-openacc.cu
cufft-multi: cufft-multi.cu
#	nvcc -O2 -arch=sm_20 -lcufft -o cufft-multi cufft-multi.cu
	nvcc -g -G -lcufft -o cufft-multi cufft-multi.cu
cufft: cufft.cu
#	nvcc -O2 -arch=sm_20 -lcufft -o cufft cufft.cu
	nvcc -g -G -lcufft -o cufft cufft.cu
cusparse: cusparse.cu
#	nvcc -O2 -arch=sm_20 -lcusparse -o cusparse cusparse.cu
	nvcc -g -G -lcusparse -o cusparse cusparse.cu
rand-kernel: rand-kernel.cu
#	nvcc -O2 -arch=sm_20 -lcurand -o rand-kernel rand-kernel.cu
	nvcc -g -G -lcurand -o rand-kernel rand-kernel.cu
replace-rand-streams: replace-rand-streams.cu
#	nvcc -O2 -arch=sm_20 -lcurand -o replace-rand-streams replace-rand-streams.cu
	nvcc -g -G -lcurand -o replace-rand-streams replace-rand-streams.cu
replace-rand: replace-rand.cu
#	nvcc -O2 -arch=sm_20 -lcurand -o replace-rand replace-rand.cu
	nvcc -g -G -lcurand -o replace-rand replace-rand.cu
%: %.cu
#	nvcc -O2 -arch=sm_20 -o $@ $<
	nvcc -g -G -o $@ $<
%: %.c
#	gcc -O2 -std=c99 -o $@ $<
	gcc -g -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
