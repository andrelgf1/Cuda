APPS=hello threads threads_2

all: ${APPS}

%: %.cu
#	nvcc -O2 -arch=sm_20 -o $@ $<
	nvcc  -g -G -o $@ $<

%: %.c
	g++ -pthread -g -o $@ $<

clean:
	rm -f ${APPS}
