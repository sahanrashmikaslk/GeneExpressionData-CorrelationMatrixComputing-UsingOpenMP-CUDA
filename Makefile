# Compiler definitions
CC = gcc
NVCC = nvcc

# GPU Architecture to target. Your MX230 is a Pascal GPU, which is sm_61.
# THIS IS THE KEY FIX for your specific hardware.
ARCH_FLAG = -arch=sm_61

# Compiler flags
CFLAGS = -O3
OMPFLAGS = -fopenmp
# Pass the -fopenmp flag to the host compiler used by nvcc
CUDA_HOST_FLAGS = -Xcompiler "-fopenmp"
# Link the math and OpenMP libraries
LIBS = -lm -lgomp

# Target executables
BASIC_TARGETS = serial openmp verify_accuracy
CUDA_TARGETS = cuda hybrid
TARGETS = $(BASIC_TARGETS)

# Check if CUDA is available
NVCC_EXISTS := $(shell command -v nvcc 2> /dev/null)
ifdef NVCC_EXISTS
    TARGETS += $(CUDA_TARGETS)
endif

all: $(TARGETS)
	@echo "Built targets: $(TARGETS)"

serial: src/corr_serial.c
	$(CC) $(CFLAGS) -o $@ $< -lm

openmp: src/corr_openmp.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -o $@ $< -lm

# The cuda target needs the correct ARCH_FLAG for your GPU
cuda: src/corr_cuda.cu
	$(NVCC) $(CFLAGS) $(ARCH_FLAG) $(CUDA_HOST_FLAGS) -o $@ $< $(LIBS)

# The hybrid target also needs the correct ARCH_FLAG
hybrid: src/corr_hybrid.cu
	$(NVCC) $(CFLAGS) $(ARCH_FLAG) $(CUDA_HOST_FLAGS) -o $@ $< $(LIBS)

# Accuracy verification tool
verify_accuracy: verify_accuracy.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f $(BASIC_TARGETS) $(CUDA_TARGETS)

# Separate targets for CPU-only builds
cpu-only: $(BASIC_TARGETS)
	@echo "✅ CPU-only build completed"

# Help target
# help:
# 	@echo "Available targets:"
# 	@echo "  all       - Build all available implementations"
# 	@echo "  cpu-only  - Build only CPU implementations (serial, openmp)"
# 	@echo "  serial    - Build serial implementation"
# 	@echo "  openmp    - Build OpenMP implementation"
# 	@echo "  cuda      - Build CUDA implementation (requires nvcc)"
# 	@echo "  hybrid    - Build hybrid implementation (requires nvcc)"
# 	@echo "  verify_accuracy - Build accuracy verification tool"
# 	@echo "  clean     - Remove all executables"