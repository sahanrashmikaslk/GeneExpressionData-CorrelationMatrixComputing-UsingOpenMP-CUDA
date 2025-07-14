#!/bin/bash

# Quick Setup and Test Script for WSL Plotting

echo "=== WSL Plotting Setup and Test ==="
echo ""

# Set matplotlib backend for WSL
export MPLBACKEND=Agg
echo "✅ Set matplotlib backend to Agg (non-interactive)"

# Test Python packages
echo ""
echo "Testing Python packages..."
python3 -c "
import pandas, matplotlib, seaborn, numpy
matplotlib.use('Agg')
print('✅ All packages imported successfully')
print('✅ Matplotlib backend:', matplotlib.get_backend())
"

# Create a simple test plot
echo ""
echo "Creating test plot..."
cat > test_plot.py << 'EOF'
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import numpy as np

# Create test data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, 'b-', label='Serial (baseline)', linewidth=2)
plt.plot(x, y2, 'g-', label='OpenMP', linewidth=2)
plt.plot(x, y3, 'r-', label='CUDA', linewidth=2)

plt.title('WSL Plotting Test - Sample Performance Curves')
plt.xlabel('Problem Size')
plt.ylabel('Execution Time (seconds)')
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot
plt.savefig('benchmarks/results/test_plot.png', dpi=300, bbox_inches='tight')
print('✅ Test plot saved to benchmarks/results/test_plot.png')
plt.close()
EOF

# Create results directory if it doesn't exist
mkdir -p benchmarks/results

# Run the test
python3 test_plot.py

# Check if plot was created
if [ -f "benchmarks/results/test_plot.png" ]; then
    echo "✅ Test plot successfully created"
    echo "   File size: $(ls -lh benchmarks/results/test_plot.png | awk '{print $5}')"
else
    echo "❌ Test plot creation failed"
fi

# Clean up test file
rm -f test_plot.py

echo ""
echo "=== Ready to run benchmarks and generate plots! ==="
echo ""
echo "Next steps:"
echo "1. Build executables: make clean && make all"
echo "2. Run quick benchmark: ./benchmarks/quick_benchmark.sh"
echo "3. Generate plots: python3 benchmarks/plot_results.py benchmarks/results/quick_benchmark_*.csv"
echo ""
echo "Or run everything at once with: ./example_usage.sh"
