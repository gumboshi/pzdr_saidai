
#useCUDA := true
NUM_BLOCK := 52 # for K20c
#NUM_BLOCK = 56 # for K20X

CUDATOP := /usr/local/cuda/
NVCC := nvcc
NVFLAGS := -O3 -arch=sm_35 -DNUM_BLOCK=$(NUM_BLOCK) -I.

# CC := icc
# CPP := icpp
# CFLAGS := -O3 -openmp -I.
# LFLAGS := -openmp

CC := gcc
CPP := g++
CFLAGS := -O3 -fopenmp -I.
LFLAGS := -fopenmp -lm
# CFLAGS := -O3 -I.
# LFLAGS := -lm

#ifeq ($(useCUDA),true)
CUDACFLAGS := $(CFLAGS) -DCUDA 
CUDALFLAGS := $(LFLAGS) -L $(CUDATOP)/lib64 -lcudart 
#endif

TARGET := pzdr_saidai.exe

all: pzdr_saidai.o
	${CC} ${LFLAGS} $^ -o ${TARGET}

gpu: pzdr_saidai_dcuda.o pzdr_saidai_cuda.o
	${CPP} ${CUDALFLAGS} $^ -o ${TARGET}

pzdr_saidai_cuda.o : pzdr_saidai.cu
	${NVCC} -c ${NVFLAGS} $< -o $@

pzdr_saidai.o : pzdr_saidai.c
	${CC} -c ${CFLAGS} $< -o $@

pzdr_saidai_dcuda.o : pzdr_saidai.c
	${CC} -c ${CUDACFLAGS} $< -o $@

clean:
	rm -f *.o *~ *.exe