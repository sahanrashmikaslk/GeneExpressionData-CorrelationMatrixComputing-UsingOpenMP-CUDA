#!/bin/bash

# Accuracy Verification Script for Correlation Matrix Computing
# This script compiles and runs RMSE verification for all implementations

set -e

# Colors for output
# RED='\033[0;31m'
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# NC='\033[0m' # No Color

echo -e "$ ║                CORRELATION MATRIX ACCURACY VERIFICATION SUITE                ║$ "
echo ""

# Check if we're in the right directory
if [ ! -f "Makefile" ]; then
    echo -e "$ Error: Makefile not found. Please run this script from the project root directory.$ "
    exit 1
fi

# Clean and compile all implementations
echo -e "$ Compiling all implementations...$"
make clean

# Check if CUDA is available
if command -v nvcc >/dev/null 2>&1; then
    echo -e "$ CUDA detected - building all implementations$"
    make all
else
    echo -e "$ CUDA not detected - building CPU-only implementations$"
    make cpu-only
fi

if [ $? -ne 0 ]; then
    echo -e "$ Compilation failed. Please check for errors.$"
    exit 1
fi

echo -e "$ All implementations compiled successfully!$"
echo ""

# Create results directory if it doesn't exist
mkdir -p benchmarks/results

# Run accuracy verification
echo -e "$ Running accuracy verification tests...$"
echo ""

./verify_accuracy

echo ""

