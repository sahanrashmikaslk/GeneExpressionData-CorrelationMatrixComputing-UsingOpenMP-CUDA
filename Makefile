# Compiler 

#definitions
CC = gcc
NVCC = nvcc

# flags
CFLAGS = -O3
OMPFLAGS = -fopenmp

# Pass the -fopenmp flag to the host compiler used by nvcc
CUDA_HOST_FLAGS = -Xcompiler "-fopenmp"

# Link the math and OpenMP libraries
LIBS = -lm -lgomp

# Suffix rule for .cu files
.SUFFIXES: .c .cu

# Target executables
TARGETS = serial openmp cuda

all: $(TARGETS)


serial: src/corr_serial.c
	$(CC) $(CFLAGS) -o $@ $< -lm


openmp: src/corr_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< -lm


cuda: src/corr_cuda.cu
	$(NVCC) $(CFLAGS) $(CUDA_HOST_FLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS)