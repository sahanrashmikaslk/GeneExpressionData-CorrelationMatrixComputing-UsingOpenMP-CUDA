# Large Pearson Correlation Matrix Computing Using Hybrid OpenMP and CUDA

**EC7207 - High Performance Computing Project**  
**Student:** Lelwala L.G.S.R (EG/2020/4047)

## Overview

This project implements a high-performance hybrid parallel algorithm for computing large-scale Pearson correlation matrices, combining CPU multi-threading (OpenMP) and GPU massive parallelism (CUDA) to achieve significant computational speedup.

## Table of Contents

- [Prerequisites](#prerequisites)
- [WSL Environment Setup](#wsl-environment-setup)
- [Project Structure](#project-structure)
- [Implementation Roadmap](#implementation-roadmap)
- [Building and Running](#building-and-running)
- [Performance Testing](#performance-testing)
- [Results](#results)

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support (Compute Capability 3.0+)
- Multi-core CPU
- Minimum 8GB RAM (16GB+ recommended for large datasets)

### Software Requirements
- Windows 10/11 with WSL2
- Ubuntu 20.04+ in WSL
- NVIDIA GPU drivers for Windows
- CUDA Toolkit 11.0+
- GCC compiler with OpenMP support

## WSL Environment Setup

### 1. Install WSL2 and Ubuntu
```bash
# In Windows PowerShell (as Administrator)
wsl --install -d Ubuntu-20.04
```

### 2. Install CUDA in WSL
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build essentials
sudo apt install -y build-essential

# Download and install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 3. Verify Installation
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check OpenMP support
gcc -fopenmp --version
```

## Project Structure

```
pearson-correlation-hpc/
├── src/
│   ├── serial.c                 # Sequential implementation
│   ├── openmp.c                 # OpenMP parallel implementation
│   ├── cuda.cu                  # CUDA implementation
│   ├── hybrid.cu                # Hybrid OpenMP + CUDA implementation
│   └── utils.h                  # Common utilities and data structures
├── data/
│   ├── generate_data.c          # Test data generator
│   └── sample_datasets/         # Sample test datasets
├── benchmarks/
│   ├── benchmark.sh             # Automated benchmarking script
│   └── results/                 # Performance results
├── Makefile                     # Build configuration
└── README.md                    # This file
```

## Implementation Roadmap

### Phase 1: Sequential Implementation (Baseline)
**File:** `src/serial.c`

**Objectives:**
- Implement basic Pearson correlation matrix computation
- Establish performance baseline
- Validate correctness of correlation calculations

**Key Components:**
```c
// Pearson correlation coefficient formula
double pearson_correlation(double *x, double *y, int n);

// Sequential matrix computation
void compute_correlation_matrix_serial(double **data, int n_vars, int n_samples, double **result);
```

### Phase 2: OpenMP Implementation
**File:** `src/openmp.c`

**Objectives:**
- Parallelize correlation computation across CPU cores
- Implement efficient work distribution
- Optimize memory access patterns

**Key Features:**
- Parallel for loops with optimal scheduling
- Thread-safe correlation computation
- Memory optimization for cache efficiency

### Phase 3: CUDA Implementation
**File:** `src/cuda.cu`

**Objectives:**
- Leverage GPU massive parallelism
- Implement efficient CUDA kernels
- Optimize memory transfers and shared memory usage

**Key Components:**
```cuda
// CUDA kernel for correlation computation
__global__ void compute_correlations_kernel(float *data, float *result, int n_vars, int n_samples);

// Shared memory optimization
__shared__ float shared_data[TILE_SIZE][TILE_SIZE];
```

### Phase 4: Hybrid OpenMP + CUDA Implementation
**File:** `src/hybrid.cu`

**Objectives:**
- Combine CPU and GPU parallelism
- Implement efficient work distribution between CPU and GPU
- Optimize data transfers and synchronization

**Strategy:**
- CPU handles data partitioning and coordination
- GPU processes large correlation tiles
- Asynchronous execution and memory transfers

## Building and Running

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd pearson-correlation-hpc
mkdir -p data/sample_datasets benchmarks/results
```

### 2. Generate Test Data
```bash
# Compile data generator
gcc -o data/generate_data data/generate_data.c -lm

# Generate test datasets of different sizes
./data/generate_data 100 1000 > data/sample_datasets/small.csv    # 100 vars, 1000 samples
./data/generate_data 500 5000 > data/sample_datasets/medium.csv   # 500 vars, 5000 samples
./data/generate_data 1000 10000 > data/sample_datasets/large.csv  # 1000 vars, 10000 samples
```

### 3. Build All Implementations
```bash
# Build all versions
make all

# Or build individually
make serial    # Sequential version
make openmp    # OpenMP version
make cuda      # CUDA version
make hybrid    # Hybrid version
```

### 4. Run Implementations Progressively
```bash
# 1. Run serial implementation (baseline)
echo "Running Serial Implementation..."
time ./bin/serial data/sample_datasets/small.csv

# 2. Run OpenMP implementation
echo "Running OpenMP Implementation..."
export OMP_NUM_THREADS=4
time ./bin/openmp data/sample_datasets/small.csv

# 3. Run CUDA implementation
echo "Running CUDA Implementation..."
time ./bin/cuda data/sample_datasets/small.csv
  
# 4. Run Hybrid implementation
echo "Running Hybrid Implementation..."
export OMP_NUM_THREADS=4
time ./bin/hybrid data/sample_datasets/small.csv
```

## Performance Testing

### Automated Benchmarking
```bash
# Run comprehensive benchmarks
chmod +x benchmarks/benchmark.sh
./benchmarks/benchmark.sh

# This will test all implementations with different:
# - Dataset sizes (100x1000, 500x5000, 1000x10000, 2000x20000)
# - Thread counts (1, 2, 4, 8, 16)
# - GPU configurations
```

### Manual Performance Testing
```bash
# Test scalability with different thread counts
for threads in 1 2 4 8 16; do
    export OMP_NUM_THREADS=$threads
    echo "Testing with $threads threads"
    time ./bin/openmp data/sample_datasets/medium.csv
done

# Test different dataset sizes
for dataset in small medium large; do
    echo "Testing dataset: $dataset"
    time ./bin/hybrid data/sample_datasets/$dataset.csv
done
```

## Expected Performance Improvements

| Implementation | Expected Speedup | Best Use Case |
|---------------|------------------|---------------|
| Serial | 1.0x (baseline) | Small datasets, validation |
| OpenMP | 2-8x | Medium datasets, CPU-bound |
| CUDA | 10-100x | Large datasets, GPU-optimized |
| Hybrid | 15-150x | Very large datasets, optimal resource usage |

## Key Performance Metrics

1. **Execution Time**: Total computation time
2. **Speedup**: Performance gain over serial implementation
3. **Efficiency**: Speedup per processing unit
4. **Scalability**: Performance with increasing dataset size
5. **Memory Usage**: Peak memory consumption

## Troubleshooting

### Common Issues

**CUDA Not Found:**
```bash
# Check CUDA installation
which nvcc
echo $PATH | grep cuda
```

**OpenMP Not Working:**
```bash
# Verify OpenMP support
echo | cpp -fopenmp -dM | grep -i openmp
```

**Memory Issues:**
```bash
# Monitor memory usage
watch -n 1 'free -h && nvidia-smi'
```

## Results Analysis

After running all implementations, analyze:

1. **Performance Scaling**: How each implementation performs with increasing data size
2. **Resource Utilization**: CPU vs GPU efficiency
3. **Memory Patterns**: Memory bandwidth and access patterns
4. **Hybrid Benefits**: Advantages of combining OpenMP and CUDA

## Future Enhancements

- [ ] Multi-GPU support
- [ ] Distributed computing with MPI
- [ ] Mixed precision computation
- [ ] Adaptive work distribution
- [ ] Memory-mapped file I/O for large datasets

## References

- CUDA Programming Guide
- OpenMP Specification
- Pearson Correlation Coefficient Mathematics
- High-Performance Computing Best Practices

## License

This project is for educational purposes as part of the EC7207 High Performance Computing course.
