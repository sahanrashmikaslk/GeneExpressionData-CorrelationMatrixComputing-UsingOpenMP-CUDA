#!/bin/bash

# Simple RMSE Test for CPU implementations
# Tests accuracy between serial and OpenMP implementations

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   CPU IMPLEMENTATIONS RMSE TEST                  ║${NC}"
echo -e "${BLUE}║                    EE7218/EC7207 Project                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to calculate RMSE between two output files
calculate_rmse() {
    local file1=$1
    local file2=$2
    local N=$3
    
    # Extract correlation matrices from output files and calculate RMSE
    python3 -c "
import sys
import math
import re

def extract_matrix(filename, N):
    matrix = []
    with open(filename, 'r') as f:
        content = f.read()
        # Find the correlation matrix section
        lines = content.split('\n')
        in_matrix = False
        for line in lines:
            if 'Output Correlation Matrix' in line:
                in_matrix = True
                continue
            if in_matrix and '-------' in line:
                break
            if in_matrix and line.strip():
                # Extract numbers from line
                numbers = re.findall(r'-?\d+\.\d+', line)
                if numbers:
                    matrix.extend([float(x) for x in numbers[:N]])
    return matrix[:N*N]

def calculate_rmse(matrix1, matrix2):
    if len(matrix1) != len(matrix2):
        return float('inf')
    
    sum_sq_diff = sum((a - b)**2 for a, b in zip(matrix1, matrix2))
    return math.sqrt(sum_sq_diff / len(matrix1))

N = int('$N')
try:
    matrix1 = extract_matrix('$file1', N)
    matrix2 = extract_matrix('$file2', N)
    rmse = calculate_rmse(matrix1, matrix2)
    print(f'{rmse:.2e}')
except Exception as e:
    print('ERROR')
"
}

# Test configurations
declare -a TEST_SIZES=("50 100" "100 200" "200 500")

# Compile implementations
echo -e "${YELLOW}🔨 Compiling CPU implementations...${NC}"
make clean
make cpu-only

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Compilation successful!${NC}"
echo ""

# Create temp directory
mkdir -p /tmp/rmse_test
cd /tmp/rmse_test

echo -e "${GREEN}🧪 Running RMSE accuracy tests:${NC}"
echo "═══════════════════════════════════════════════════════════════════"
printf "%-15s %-15s %-15s %-15s %s\n" "Matrix Size" "Serial Time" "OpenMP Time" "RMSE" "Accuracy"
echo "───────────────────────────────────────────────────────────────────"

for test_config in "${TEST_SIZES[@]}"; do
    read -r N M <<< "$test_config"
    
    # Run serial
    timeout 30s ${OLDPWD}/serial $N $M > serial_output.txt 2>&1
    serial_exit=$?
    
    # Run OpenMP  
    timeout 30s ${OLDPWD}/openmp $N $M > openmp_output.txt 2>&1
    openmp_exit=$?
    
    if [ $serial_exit -eq 0 ] && [ $openmp_exit -eq 0 ]; then
        # Extract execution times
        serial_time=$(grep "Execution Time:" serial_output.txt | awk '{print $3}')
        openmp_time=$(grep "Execution Time:" openmp_output.txt | awk '{print $3}')
        
        # Calculate RMSE
        rmse=$(calculate_rmse "serial_output.txt" "openmp_output.txt" $N)
        
        # Determine accuracy level
        if [[ "$rmse" == "ERROR" ]]; then
            accuracy="❌ ERROR"
        elif (( $(echo "$rmse < 1e-12" | bc -l) )); then
            accuracy="✅ EXCELLENT"
        elif (( $(echo "$rmse < 1e-6" | bc -l) )); then
            accuracy="✅ VERY GOOD"
        elif (( $(echo "$rmse < 1e-3" | bc -l) )); then
            accuracy="⚠️  ACCEPTABLE"
        else
            accuracy="❌ POOR"
        fi
        
        printf "%-15s %-15s %-15s %-15s %s\n" "${N}×${M}" "${serial_time}s" "${openmp_time}s" "$rmse" "$accuracy"
    else
        printf "%-15s %-15s %-15s %-15s %s\n" "${N}×${M}" "FAILED" "FAILED" "N/A" "❌ EXEC ERROR"
    fi
done

echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo -e "${YELLOW}📋 RMSE Accuracy Criteria:${NC}"
echo "   ✅ EXCELLENT:  RMSE < 1e-12"
echo "   ✅ VERY GOOD:  RMSE < 1e-6"
echo "   ⚠️  ACCEPTABLE: RMSE < 1e-3"
echo "   ❌ POOR:       RMSE ≥ 1e-3"
echo ""

# Cleanup
cd ${OLDPWD}
rm -rf /tmp/rmse_test

echo -e "${GREEN}🎉 CPU RMSE testing completed!${NC}"

# Show speedup information
echo ""
echo -e "${BLUE}💡 For your project report, note:${NC}"
echo "   • Serial implementation serves as the reference baseline"
echo "   • OpenMP should show identical results (RMSE ≈ 0)"
echo "   • Any RMSE > 1e-6 indicates potential numerical precision issues"
echo "   • OpenMP speedup = Serial_time / OpenMP_time"
