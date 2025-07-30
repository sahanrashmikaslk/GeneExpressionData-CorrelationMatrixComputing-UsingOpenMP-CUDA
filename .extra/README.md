# Large Pearson Correlation Matrix Computing Using Hybrid OpenMP and CUDA

**EC7207 - High Performance Computing Project**  
**Student:** Lelwala L.G.S.R (EG/2020/4047)

## Overview

This project implements a high-performance hybrid parallel algorithm for computing large-scale Pearson correlation matrices, combining CPU multi-threading (OpenMP) and GPU massive parallelism (CUDA) to achieve significant computational speedup.

## Table of Contents

- [Prerequisites](#prerequisites)
- [WSL Environment Setup](#wsl-environment-setup)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Building and Running](#building-and-running)
- [Benchmarking and Performance Analysis](#benchmarking-and-performance-analysis)
- [Performance Results](#performance-results)
- [Troubleshooting](#troubleshooting)
- [Project Deliverables](#project-deliverables)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

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
- Python 3.8+ with data analysis libraries (for benchmarking)

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
GeneExpressionData-CorrelationMatrixComputing-UsingOpenMP-CUDA/
├── src/
│   ├── corr_serial.c            # Sequential baseline implementation
│   ├── corr_openmp.c            # OpenMP parallel implementation
│   ├── corr_cuda.cu             # CUDA GPU implementation
│   └── corr_hybrid.cu           # Hybrid OpenMP + CUDA implementation
├── benchmarks/
│   ├── benchmark.sh             # Comprehensive automated benchmarking
│   ├── quick_benchmark.sh       # Quick validation benchmark
│   ├── plot_results.py          # Performance analysis and plotting
│   ├── requirements.txt         # Python dependencies
│   └── results/                 # Performance results and plots
├── Makefile                     # Build configuration
└── README.md                    # This file
```

## Implementation Details

### Pearson Correlation Formula

The Pearson correlation coefficient between two variables X and Y is computed as:

```
r(X,Y) = (n∑XY - ∑X∑Y) / √[(n∑X² - (∑X)²)(n∑Y² - (∑Y)²)]
```

Where:

- n = number of samples
- ∑XY = sum of products of paired samples
- ∑X, ∑Y = sum of individual variable samples
- ∑X², ∑Y² = sum of squared samples

### Implementation Approaches

#### 1. Serial Implementation (`corr_serial.c`)

- **Purpose:** Baseline performance measurement
- **Algorithm:** Nested loops computing correlation for each pair (i,j)
- **Complexity:** O(N²M) where N = variables, M = samples
- **Features:**
  - Simple and straightforward implementation
  - Computes only upper triangle (symmetric matrix)
  - High precision timing using `clock_gettime()`

#### 2. OpenMP Implementation (`corr_openmp.c`)

- **Purpose:** CPU parallelization using multiple threads
- **Algorithm:** Parallelized outer loop with dynamic scheduling
- **Features:**
  - `#pragma omp parallel for schedule(dynamic)`
  - Load balancing across CPU cores
  - Thread-safe correlation computation
  - Automatic scaling with available CPU cores

#### 3. CUDA Implementation (`corr_cuda.cu`)

- **Purpose:** GPU massive parallelism
- **Algorithm:** 2D grid of CUDA threads, each computing one correlation
- **Features:**
  - Optimized memory access patterns
  - Shared memory utilization for better performance
  - Coalesced global memory access
  - Multiple kernel variants (basic and optimized)

#### 4. Hybrid Implementation (`corr_hybrid.cu`)

- **Purpose:** Combine CPU preprocessing with GPU computation
- **Algorithm:** CPU manages data preparation, GPU handles correlations
- **Features:**
  - OpenMP for parallel data initialization
  - CUDA streams for asynchronous execution
  - Optimized data transfer patterns
  - Adaptive workload distribution

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
time ./bin/hybrid data/sample_datasets/small.csv
```

## Performance Results

### Current Benchmark Results (N=1000, M=2000)

Based on your recent test results:

| Implementation     | Execution Time | Speedup | Performance Analysis       |
| ------------------ | -------------- | ------- | -------------------------- |
| Serial             | 1.321s         | 1.00x   | Baseline performance       |
| OpenMP (8 threads) | 0.611s         | 2.16x   | Good CPU parallelization   |
| CUDA (MX230)       | 0.218s         | 6.06x   | Excellent GPU acceleration |
| Hybrid             | 0.745s         | 1.77x   | **Needs optimization**     |

### Key Observations

#### ✅ Successes

- **CUDA Implementation:** Achieves 6x speedup, demonstrating effective GPU utilization
- **OpenMP Implementation:** Good 2.16x speedup on 8-core CPU
- **Serial Baseline:** Stable and reliable baseline implementation

#### ⚠️ Areas for Improvement

- **Hybrid Implementation:** Currently underperforming at 1.77x speedup
  - Should theoretically outperform individual implementations
  - Likely issues: overhead from CPU-GPU coordination, suboptimal work distribution
  - Recent optimization attempts focused on reducing this overhead

#### 🎯 Performance Goals

- **Target Hybrid Speedup:** 8-12x (combining strengths of CPU and GPU)
- **Optimization Focus:** Reduce data transfer overhead, improve work distribution
- **Scalability Target:** Better performance on larger problem sizes (N>2000)

### Hardware Configuration

- **CPU:** 8 cores with OpenMP support
- **GPU:** NVIDIA GeForce MX230 (Pascal architecture, Compute Capability 6.1)
- **Memory:** 4GB GPU memory, sufficient system RAM
- **CUDA Version:** Compatible with sm_61 architecture

### Next Steps for Optimization

1. **Profile GPU utilization** during hybrid execution
2. **Optimize data transfer patterns** between CPU and GPU
3. **Implement better work partitioning** strategies
4. **Test with larger datasets** to find optimal problem sizes
5. **Consider asynchronous execution** patterns for better overlap

## Troubleshooting

### Common Build Issues

#### CUDA Architecture Mismatch

```bash
# Error: No kernel image is available for execution on the device
# Solution: Check your GPU compute capability
nvidia-smi

# Update Makefile ARCH_FLAG for your specific GPU
# For MX230 (Pascal): -arch=sm_61
# For RTX series: -arch=sm_75 or higher
```

#### OpenMP Not Found

```bash
# Error: fatal error: omp.h: No such file or directory
# Solution: Install OpenMP development libraries
sudo apt install libomp-dev

# Verify installation
echo '#include <omp.h>' | gcc -fopenmp -x c -
```

#### CUDA Toolkit Issues

```bash
# Error: nvcc: command not found
# Solution: Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
nvidia-smi
```

### Performance Troubleshooting

#### Poor GPU Performance

1. **Check GPU Memory:**

   ```bash
   nvidia-smi  # Monitor GPU memory during execution
   ```

2. **Verify CUDA Architecture:**

   ```bash
   # Ensure Makefile uses correct architecture flag
   grep ARCH_FLAG Makefile
   ```

3. **Monitor GPU Utilization:**
   ```bash
   # Run in another terminal during benchmark
   watch -n 1 nvidia-smi
   ```

#### OpenMP Performance Issues

1. **Check Thread Affinity:**

   ```bash
   export OMP_PROC_BIND=true
   export OMP_PLACES=cores
   ```

2. **Verify Thread Count:**
   ```bash
   export OMP_NUM_THREADS=$(nproc)
   echo $OMP_NUM_THREADS
   ```

#### Hybrid Implementation Optimization

1. **Profile Execution:**

   ```bash
   # Use nvprof for CUDA profiling
   nvprof ./hybrid 1000 2000
   ```

2. **Memory Transfer Analysis:**
   ```bash
   # Monitor data transfer times
   nvprof --print-gpu-trace ./hybrid 1000 2000
   ```

### System Requirements Verification

```bash
# Check system specifications
echo "=== System Information ==="
echo "CPU Cores: $(nproc)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
echo "CUDA Version: $(nvcc --version | grep 'release' | awk '{print $6}')"
echo "OpenMP Max Threads: $(echo '#include <omp.h>
#include <stdio.h>
int main() { printf("%d\n", omp_get_max_threads()); return 0; }' | gcc -fopenmp -x c - -o /tmp/omp_test && /tmp/omp_test && rm -f /tmp/omp_test)"
```

## Project Deliverables

### Code Components

- ✅ Serial baseline implementation (`corr_serial.c`)
- ✅ OpenMP parallel implementation (`corr_openmp.c`)
- ✅ CUDA GPU implementation (`corr_cuda.cu`)
- ✅ Hybrid OpenMP+CUDA implementation (`corr_hybrid.cu`)
- ✅ Comprehensive build system (`Makefile`)

### Benchmarking System

- ✅ Automated benchmark suite (`benchmark.sh`)
- ✅ Quick validation testing (`quick_benchmark.sh`)
- ✅ Performance analysis and plotting (`plot_results.py`)
- ✅ Statistical analysis and reporting

### Documentation

- ✅ Comprehensive README with setup instructions
- ✅ Performance analysis and optimization guides
- ✅ Troubleshooting documentation
- ✅ Project structure and implementation details

### Expected Academic Outcomes

- **Performance Analysis:** Quantitative speedup measurements across implementations
- **Scalability Study:** Performance behavior with varying problem sizes
- **Architecture Comparison:** CPU vs GPU vs Hybrid performance characteristics
- **Optimization Research:** Investigation of hybrid parallelization strategies

## Contributing

This project is part of an academic assignment for EC7207 High Performance Computing. The implementation demonstrates:

1. **Parallel Algorithm Design:** Multiple approaches to the same computational problem
2. **Performance Engineering:** Optimization techniques for CPU and GPU architectures
3. **Hybrid Computing:** Combining multiple parallel paradigms effectively
4. **Empirical Analysis:** Comprehensive benchmarking and performance evaluation

## License

This project is developed for academic purposes as part of the EC7207 High Performance Computing course.

## Contact

**Student:** Lelwala L.G.S.R  
**Registration:** EG/2020/4047  
**Course:** EC7207 - High Performance Computing

For questions or issues related to this implementation, please refer to the troubleshooting section or consult the course materials.
