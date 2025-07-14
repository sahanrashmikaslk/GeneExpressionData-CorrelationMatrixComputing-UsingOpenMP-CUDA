#!/bin/bash

# Example Usage Script - Demonstrates the benchmarking system
# This script shows how to use the benchmark tools

echo "=== Correlation Matrix Computing - Benchmark Example ==="
echo ""

# Check if we're in the right directory
if [ ! -f "Makefile" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

echo "1. Building all implementations..."
make clean && make all

if [ $? -ne 0 ]; then
    echo "Error: Build failed. Please check your environment setup."
    exit 1
fi

echo ""
echo "2. Quick functionality test..."
echo "Testing each implementation with small matrix (100x500):"

echo "  - Serial implementation:"
./serial 100 500 | grep "Execution Time"

echo "  - OpenMP implementation:"
./openmp 100 500 | grep "Execution Time"

echo "  - CUDA implementation:"
./cuda 100 500 | grep "Execution Time"

echo "  - Hybrid implementation:"
./hybrid 100 500 | grep "Execution Time"

echo ""
echo "3. Running quick benchmark..."
./benchmarks/quick_benchmark.sh

echo ""
echo "4. Checking for Python dependencies..."
if command -v python3 &> /dev/null; then
    echo "Python3 found. Checking for required packages..."
    
    # Try to import required packages
    python3 -c "import pandas, matplotlib, seaborn, numpy" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "All Python packages available."
        
        # Find the most recent benchmark results
        latest_results=$(ls -t benchmarks/results/quick_benchmark_*.csv 2>/dev/null | head -1)
        
        if [ -n "$latest_results" ]; then
            echo ""
            echo "5. Generating performance plots..."
            python3 benchmarks/plot_results.py "$latest_results"
            echo "Plots saved to benchmarks/results/"
        else
            echo "No benchmark results found to plot."
        fi
    else
        echo "Missing Python packages. Install with:"
        echo "pip3 install -r benchmarks/requirements.txt"
    fi
else
    echo "Python3 not found. Install Python3 to generate plots."
fi

echo ""
echo "=== Example Usage Complete ==="
echo ""
echo "Next steps:"
echo "1. Review results in benchmarks/results/"
echo "2. Run full benchmark: ./benchmarks/benchmark.sh"
echo "3. Analyze performance trends with larger datasets"
echo "4. Optimize hybrid implementation based on results"
echo ""
echo "For more information, see README.md"
