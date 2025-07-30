#!/bin/bash

# Quick RMSE Testing Script
# Tests accuracy of parallel implementations against serial reference

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              QUICK RMSE ACCURACY VERIFICATION                     ║${NC}"
echo -e "${BLUE}║                 EE7218/EC7207 Project                            ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Clean any existing reference files
rm -f *.bin

# Test configuration
N=200
M=500

echo -e "${YELLOW}📊 Testing with matrix size: ${N} × ${M}${NC}"
echo ""

# Check if executables exist
if [ ! -f "serial" ] || [ ! -f "openmp" ]; then
    echo -e "${YELLOW}🔨 Compiling implementations...${NC}"
    make clean
    make serial openmp
    
    # Try CUDA if available
    if command -v nvcc >/dev/null 2>&1; then
        make cuda hybrid 2>/dev/null || echo -e "${YELLOW}⚠️  CUDA compilation skipped (GPU not available)${NC}"
    fi
fi

echo ""
echo -e "${GREEN}🚀 Running implementations:${NC}"
echo "═══════════════════════════════════════════════════════════════════"

# Run serial (creates reference)
echo -e "${BLUE}1. Running Serial Implementation (Reference)${NC}"
echo "   Command: ./serial $N $M"
./serial $N $M | grep -E "(Execution Time|ACCURACY|Implementation|Matrix Size|RMSE|Accuracy)"
echo ""

# Run OpenMP
echo -e "${BLUE}2. Running OpenMP Implementation${NC}"
echo "   Command: ./openmp $N $M"
./openmp $N $M | grep -E "(Execution Time|ACCURACY|Implementation|Matrix Size|RMSE|Accuracy|Num threads)"
echo ""

# Run CUDA if available
if [ -f "cuda" ]; then
    echo -e "${BLUE}3. Running CUDA Implementation${NC}"
    echo "   Command: ./cuda $N $M"
    ./cuda $N $M | grep -E "(Execution Time|ACCURACY|Implementation|Matrix Size|RMSE|Accuracy|Using GPU)" 2>/dev/null || echo -e "${YELLOW}   ⚠️  CUDA execution skipped (GPU not available)${NC}"
    echo ""
fi

# Run Hybrid if available
if [ -f "hybrid" ]; then
    echo -e "${BLUE}4. Running Hybrid Implementation${NC}"
    echo "   Command: ./hybrid $N $M"
    ./hybrid $N $M | grep -E "(Execution Time|ACCURACY|Implementation|Matrix Size|RMSE|Accuracy|OpenMP threads)" 2>/dev/null || echo -e "${YELLOW}   ⚠️  Hybrid execution skipped (GPU not available)${NC}"
    echo ""
fi

echo "═══════════════════════════════════════════════════════════════════"
echo -e "${GREEN}✅ Testing completed!${NC}"
echo ""
echo -e "${YELLOW}📋 RMSE Interpretation:${NC}"
echo "   • RMSE < 1e-12: ✅ EXCELLENT accuracy"
echo "   • RMSE < 1e-6:  ✅ VERY GOOD accuracy"  
echo "   • RMSE < 1e-3:  ⚠️  ACCEPTABLE accuracy"
echo "   • RMSE ≥ 1e-3:  ❌ POOR accuracy"
echo ""
echo -e "${BLUE}💡 Tip: Run different matrix sizes by editing N and M values in this script${NC}"

# Cleanup
echo -e "${YELLOW}🧹 Cleaning up temporary files...${NC}"
rm -f *.bin
