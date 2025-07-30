# Gene Expression Data - Correlation Matrix Computing Using OpenMP & CUDA

This repository contains implementations of Pearson Correlation Matrix computation optimized for large gene expression datasets using different parallelization approaches:

- Serial (baseline) implementation
- OpenMP (multi-core CPU) parallelization
- CUDA (GPU) parallelization
- Hybrid OpenMP-CUDA implementation

## Project Structure

```
├── Makefile                   # Build system configuration
├── openmp_result.bin          # Binary result file from OpenMP implementation
├── serial_reference.bin       # Reference binary file for accuracy verification
├── serial_result.bin          # Binary result file from serial implementation
├── performance_benchmarker    # Executable for performance benchmarking
├── verify_accuracy.c          # Utility to verify accuracy between implementations
├── run_accuracy_test.sh       # Script for running accuracy tests
├── simple_rmse_test.sh        # Script for quick RMSE testing between CPU implementations
├── quick_rmse_test.sh         # Script for quick RMSE comparison
├── benchmarks/
│   ├── benchmark.sh           # Main benchmarking script
│   ├── benchmarkopenmp.c      # OpenMP benchmarking utility
│   ├── quick_benchmark.sh     # Script for quick benchmarking
│   └── results/               # Directory containing benchmark results
│       ├── benchmark_log_*.log        # Benchmark log files
│       ├── benchmark_results_*.csv    # CSV files with benchmark data
│       └── performance_results_*.csv  # Performance results data
└── src/
    ├── accuracy_utils.h       # Utilities for accuracy verification
    ├── corr_serial.c          # Serial implementation
    ├── corr_openmp.c          # OpenMP implementation
    ├── corr_cuda.cu           # CUDA implementation
    └── corr_hybrid.cu         # Hybrid OpenMP-CUDA implementation
```

## Implementations

### Serial Implementation (src/corr_serial.c)

The baseline implementation computes the Pearson correlation matrix using a serial approach. This serves as the reference for accuracy verification.

### OpenMP Implementation (src/corr_openmp.c)

Parallel implementation using OpenMP for multi-core CPU computation. This implementation parallelizes the outer loop of correlation calculation.

### CUDA Implementation (src/corr_cuda.cu)

GPU-accelerated implementation using CUDA, which offloads correlation calculation to NVIDIA GPUs. The implementation uses a 2D thread grid to calculate correlation coefficients in parallel.

### Hybrid Implementation (src/corr_hybrid.cu)

A combined approach that uses both OpenMP and CUDA to leverage both CPU and GPU resources.

## Building the Project

The project uses a Makefile build system that automatically detects available compilers and hardware:

```bash
# Build all implementations (requires GCC for CPU and NVCC for GPU)
make

# Build only CPU implementations (serial and OpenMP)
make cpu-only

# Build individual implementations
make serial
make openmp
make cuda
make hybrid

# Clean all build artifacts
make clean
```

## Running the Implementations

All implementations follow the same command-line interface:

```bash
./serial <N_variables> <M_samples>
./openmp <N_variables> <M_samples>
./cuda <N_variables> <M_samples>
./hybrid <N_variables> <M_samples>
```

Where:

- `N_variables`: Number of variables (genes) in the correlation matrix
- `M_samples`: Number of samples for each variable (gene)

## Benchmarking

The repository includes comprehensive benchmarking tools:

```bash
# Run full benchmark suite (tests all implementations with different sizes)
./benchmarks/benchmark.sh

# Run quick benchmark
./benchmarks/quick_benchmark.sh
```

Benchmark results are stored in CSV files in the `benchmarks/results/` directory.

## Accuracy Verification

To verify the accuracy of different implementations against the serial reference:

```bash
# Run the accuracy verification
./run_accuracy_test.sh

# Run a quick RMSE test between CPU implementations
./simple_rmse_test.sh
```

The accuracy is measured using Root Mean Square Error (RMSE) between the computed correlation matrices.

## Performance Metrics

Performance metrics are collected during benchmarks:

- Execution time (in seconds)
- Speedup (relative to serial implementation)
- RMSE (accuracy compared to serial implementation)

## Hardware Requirements

- **CPU:** Any multi-core CPU supporting OpenMP
- **GPU:** NVIDIA GPU with CUDA support (Compute Capability 6.1 or higher)
- **Memory:** Sufficient RAM to hold the input and output matrices (scales with N and M)

## Software Requirements

- GCC compiler with OpenMP support
- NVIDIA CUDA Toolkit (for GPU implementations)


## Acknowledgements

This project was developed as part of a high-performance computing course to demonstrate parallel programming techniques for scientific computing applications.
