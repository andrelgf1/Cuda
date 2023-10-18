CU_APPS=nestedHelloWorld nestedReduce nestedReduce2 nestedReduceNosync \
	    reduceInteger simpleDeviceQuery simpleDivergence sumMatrix simpleDivergence2 \
		sumMatrix2
C_APPS=

# all: ${C_APPS} ${CU_APPS}

# %: %.cu
# 	nvcc -O2 -arch=sm_35 -o $@ $< -lcudadevrt --relocatable-device-code true
# %: %.c
# 	gcc -O2 -std=c99 -o $@ $<
# clean:
# 	rm -f ${CU_APPS} ${C_APPS}

all: ${C_APPS} ${CU_APPS}

%: %.cu
#   nvcc -O2 -o $@ $< -lcudadevrt --relocatable-device-code true
	nvcc -g -G -o $@ $< -lcudadevrt --relocatable-device-code true
%: %.c
	gcc -g -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}	
