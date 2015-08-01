
useCUDA := true
NUM_BLOCK := 52 # for K20c
#NUM_BLOCK = 56 # for K20X

CUDATOP := /usr/local/cuda/
NVCC := nvcc
NVFLAGS := -O3 -arch=sm_35 -DNUM_BLOCK=$(NUM_BLOCK) -I.

CC := icc
CFLAGS := -O3 -openmp -I.
LFLAGS := -openmp

# CC := gcc
# CFLAGS := -O3 -fopenmp -I.

ifeq ($(useCUDA),true)
CFLAGS := $(CFLAGS) -DCUDA 
LFLAGS := $(LFLAGS) -L $(CUDATOP)/lib64 -lcudart 
endif

all: pzdr_saidai.exe

pzdr_saidai.exe : pzdr_saidai.o pzdr_saidai_cuda.o
	${CC} ${LFLAGS} $^ -o $@

pzdr_saidai_cuda.o : pzdr_saidai.cu
	${NVCC} -c ${NVFLAGS} $< -o $@

pzdr_saidai.o : pzdr_saidai.c
	${CC} -c ${CFLAGS} $< -o $@

clean:
	rm -f *.o *~ *.exe