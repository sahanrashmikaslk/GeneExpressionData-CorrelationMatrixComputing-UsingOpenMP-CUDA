#!/bin/bash

# Automated Benchmarking Script for Correlation Matrix Computing
# Author: Lelwala L.G.S.R (EG/2020/4047)
# Project: Large Pearson Correlation Matrix Computing Using Hybrid OpenMP and CUDA

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$RESULTS_DIR/benchmark_results_$TIMESTAMP.csv"
LOG_FILE="$RESULTS_DIR/benchmark_log_$TIMESTAMP.log"

# Test configurations
declare -a N_VALUES=(100 200 500 1000 1500 2000)
declare -a M_VALUES=(500 1000 1500 2000 2500 3000)
declare -a IMPLEMENTATIONS=("serial" "openmp" "cuda" "hybrid")
ITERATIONS=3  # Number of runs per test for averaging

echo -e "${BLUE}=== Correlation Matrix Computing Benchmark Suite ===${NC}"
echo -e "${BLUE}Timestamp: $(date)${NC}"
echo -e "${BLUE}Results will be saved to: $RESULTS_FILE${NC}"
echo ""

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if executables exist
check_executables() {
    print_status "Checking for required executables..."
    
    local missing=0
    for impl in "${IMPLEMENTATIONS[@]}"; do
        if [ ! -f "$PROJECT_DIR/$impl" ]; then
            print_error "Executable '$impl' not found. Please run 'make $impl' first."
            missing=1
        fi
    done
    
    if [ $missing -eq 1 ]; then
        print_error "Please build all executables before running benchmarks."
        exit 1
    fi
    
    print_status "All executables found."
}

# Function to get system information
get_system_info() {
    print_status "Gathering system information..."
    
    {
        echo "=== System Information ==="
        echo "Date: $(date)"
        echo "Hostname: $(hostname)"
        echo "OS: $(lsb_release -d 2>/dev/null | cut -f2- || echo "Unknown")"
        echo "Kernel: $(uname -r)"
        echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
        echo "CPU Cores: $(nproc)"
        echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
        
        # GPU Information
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits | head -1)"
            echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
            echo "CUDA Version: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "Unknown")"
        else
            echo "GPU: No NVIDIA GPU detected"
        fi
        
        # OpenMP Information
        echo "OpenMP Threads: $OMP_NUM_THREADS"
        echo "OpenMP Max Threads: $(echo '#include <omp.h>
#include <stdio.h>
int main() { printf("%d\n", omp_get_max_threads()); return 0; }' | gcc -fopenmp -x c - -o /tmp/omp_test && /tmp/omp_test && rm -f /tmp/omp_test)"
        
        echo "=== Benchmark Configuration ==="
        echo "N Values: ${N_VALUES[*]}"
        echo "M Values: ${M_VALUES[*]}"
        echo "Implementations: ${IMPLEMENTATIONS[*]}"
        echo "Iterations per test: $ITERATIONS"
        echo ""
    } | tee "$LOG_FILE"
}

# Function to run a single benchmark
run_benchmark() {
    local impl=$1
    local n=$2
    local m=$3
    local iteration=$4
    
    local cmd="$PROJECT_DIR/$impl $n $m"
    local output
    local execution_time
    local exit_code
    
    print_status "Running: $impl (N=$n, M=$m, iter=$iteration)"
    
    # Capture both stdout and execution time
    if output=$(timeout 300s bash -c "time $cmd 2>&1"); then
        exit_code=0
        # Extract execution time from the program output
        execution_time=$(echo "$output" | grep "Execution Time:" | awk '{print $3}' | head -1)
        
        # If execution time not found in output, extract from time command
        if [ -z "$execution_time" ]; then
            execution_time=$(echo "$output" | grep "real" | awk '{print $2}' | head -1)
        fi
        
        # If still not found, mark as error
        if [ -z "$execution_time" ]; then
            execution_time="ERROR"
            exit_code=1
        fi
    else
        exit_code=$?
        execution_time="TIMEOUT"
        if [ $exit_code -eq 124 ]; then
            print_warning "Timeout (300s) reached for $impl (N=$n, M=$m)"
        else
            print_error "Execution failed for $impl (N=$n, M=$m) with exit code $exit_code"
        fi
    fi
    
    # Log the full output
    {
        echo "=== $impl (N=$n, M=$m, iter=$iteration) ==="
        echo "$output"
        echo "Exit code: $exit_code"
        echo "Extracted time: $execution_time"
        echo ""
    } >> "$LOG_FILE"
    
    # Return the execution time
    echo "$execution_time"
}

# Function to run all benchmarks
run_all_benchmarks() {
    print_status "Starting benchmark execution..."
    
    # Create CSV header
    echo "Implementation,N,M,Iteration,ExecutionTime(s),Status" > "$RESULTS_FILE"
    
    local total_tests=$((${#IMPLEMENTATIONS[@]} * ${#N_VALUES[@]} * ${#M_VALUES[@]} * ITERATIONS))
    local current_test=0
    
    for impl in "${IMPLEMENTATIONS[@]}"; do
        print_status "Testing implementation: $impl"
        
        for n in "${N_VALUES[@]}"; do
            for m in "${M_VALUES[@]}"; do
                for iter in $(seq 1 $ITERATIONS); do
                    current_test=$((current_test + 1))
                    
                    printf "${BLUE}Progress: %d/%d (%.1f%%)${NC}\n" \
                        $current_test $total_tests \
                        $(echo "scale=1; $current_test * 100 / $total_tests" | bc -l)
                    
                    local exec_time
                    local status="SUCCESS"
                    
                    exec_time=$(run_benchmark "$impl" "$n" "$m" "$iter")
                    
                    if [[ "$exec_time" == "TIMEOUT" || "$exec_time" == "ERROR" || -z "$exec_time" ]]; then
                        status="FAILED"
                    fi
                    
                    # Write to CSV
                    echo "$impl,$n,$m,$iter,$exec_time,$status" >> "$RESULTS_FILE"
                    
                    # Small delay to prevent system overload
                    sleep 1
                done
            done
        done
    done
    
    print_status "Benchmark execution completed!"
    print_status "Results saved to: $RESULTS_FILE"
    print_status "Detailed logs saved to: $LOG_FILE"
}

# Function to generate summary statistics
generate_summary() {
    print_status "Generating benchmark summary..."
    
    local summary_file="$RESULTS_DIR/benchmark_summary_$TIMESTAMP.txt"
    
    {
        echo "=== Benchmark Summary ==="
        echo "Generated: $(date)"
        echo "Results file: $RESULTS_FILE"
        echo ""
        
        echo "=== Test Coverage ==="
        echo "Total test configurations: $((${#IMPLEMENTATIONS[@]} * ${#N_VALUES[@]} * ${#M_VALUES[@]}))"
        echo "Iterations per configuration: $ITERATIONS"
        echo "Total benchmark runs: $((${#IMPLEMENTATIONS[@]} * ${#N_VALUES[@]} * ${#M_VALUES[@]} * ITERATIONS))"
        echo ""
        
        echo "=== Success Rate by Implementation ==="
        for impl in "${IMPLEMENTATIONS[@]}"; do
            local total=$(grep "^$impl," "$RESULTS_FILE" | wc -l)
            local success=$(grep "^$impl,.*,SUCCESS$" "$RESULTS_FILE" | wc -l)
            local rate=$(echo "scale=1; $success * 100 / $total" | bc -l)
            printf "%-10s: %3d/%3d (%.1f%%)\n" "$impl" "$success" "$total" "$rate"
        done
        echo ""
        
        echo "=== Average Execution Times (seconds) ==="
        for impl in "${IMPLEMENTATIONS[@]}"; do
            echo "Implementation: $impl"
            for n in "${N_VALUES[@]}"; do
                for m in "${M_VALUES[@]}"; do
                    local avg_time=$(grep "^$impl,$n,$m,.*,SUCCESS$" "$RESULTS_FILE" | \
                        awk -F, '{sum+=$5; count++} END {if(count>0) printf "%.4f", sum/count; else print "N/A"}')
                    printf "  N=%4d, M=%4d: %8s\n" "$n" "$m" "$avg_time"
                done
            done
            echo ""
        done
        
    } | tee "$summary_file"
    
    print_status "Summary saved to: $summary_file"
}

# Main execution
main() {
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Create results directory if it doesn't exist
    mkdir -p "$RESULTS_DIR"
    
    # Set OpenMP threads if not set
    if [ -z "$OMP_NUM_THREADS" ]; then
        export OMP_NUM_THREADS=$(nproc)
        print_status "Set OMP_NUM_THREADS to $OMP_NUM_THREADS"
    fi
    
    # Check prerequisites
    check_executables
    
    # Gather system information
    get_system_info
    
    # Run benchmarks
    run_all_benchmarks
    
    # Generate summary
    generate_summary
    
    print_status "Benchmark suite completed successfully!"
    echo ""
    echo -e "${GREEN}Next steps:${NC}"
    echo "1. Review results: $RESULTS_FILE"
    echo "2. Check logs: $LOG_FILE"
    echo "3. Generate plots: python3 benchmarks/plot_results.py $RESULTS_FILE"
    echo ""
}

# Run main function
main "$@"
