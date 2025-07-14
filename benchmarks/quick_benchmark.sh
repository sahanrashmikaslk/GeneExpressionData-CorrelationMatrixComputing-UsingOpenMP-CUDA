#!/bin/bash

# Quick Benchmark Script for Testing
# Runs a smaller subset of tests for quick validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$RESULTS_DIR/quick_benchmark_$TIMESTAMP.csv"

# Quick test configurations (smaller for faster testing)
declare -a N_VALUES=(100 500 1000)
declare -a M_VALUES=(500 1000 2000)
declare -a IMPLEMENTATIONS=("serial" "openmp" "cuda" "hybrid")
ITERATIONS=2

echo "=== Quick Benchmark Test ==="
echo "This will run a smaller subset of tests for validation"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Change to project directory
cd "$PROJECT_DIR"

# Check if executables exist
for impl in "${IMPLEMENTATIONS[@]}"; do
    if [ ! -f "$PROJECT_DIR/$impl" ]; then
        echo "Error: Executable '$impl' not found. Please run 'make $impl' first."
        exit 1
    fi
done

# Create CSV header
echo "Implementation,N,M,Iteration,ExecutionTime(s),Status" > "$RESULTS_FILE"

total_tests=$((${#IMPLEMENTATIONS[@]} * ${#N_VALUES[@]} * ${#M_VALUES[@]} * ITERATIONS))
current_test=0

for impl in "${IMPLEMENTATIONS[@]}"; do
    echo "Testing implementation: $impl"
    
    for n in "${N_VALUES[@]}"; do
        for m in "${M_VALUES[@]}"; do
            for iter in $(seq 1 $ITERATIONS); do
                current_test=$((current_test + 1))
                
                printf "Progress: %d/%d (%.1f%%)\n" \
                    $current_test $total_tests \
                    $(echo "scale=1; $current_test * 100 / $total_tests" | bc -l)
                
                echo "Running: $impl (N=$n, M=$m, iter=$iter)"
                
                # Run the benchmark and capture output
                if output=$(timeout 120s ./"$impl" "$n" "$m" 2>&1); then
                    # Extract execution time
                    exec_time=$(echo "$output" | grep "Execution Time:" | awk '{print $3}' | head -1)
                    
                    if [ -n "$exec_time" ]; then
                        status="SUCCESS"
                        echo "  → Time: ${exec_time}s"
                    else
                        exec_time="ERROR"
                        status="FAILED"
                        echo "  → Failed to extract execution time"
                    fi
                else
                    exec_time="TIMEOUT"
                    status="FAILED"
                    echo "  → Timeout or execution failed"
                fi
                
                # Write to CSV
                echo "$impl,$n,$m,$iter,$exec_time,$status" >> "$RESULTS_FILE"
                
                sleep 0.5  # Small delay
            done
        done
    done
done

echo ""
echo "Quick benchmark completed!"
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "To generate plots, run:"
echo "python3 benchmarks/plot_results.py $RESULTS_FILE"
