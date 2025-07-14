# WSL Environment Setup Guide for Performance Plotting

## Prerequisites Check

First, let's verify your WSL environment is ready:

```bash
# Check Python version
python3 --version

# Check pip availability
pip3 --version

# If pip is not installed:
sudo apt update
sudo apt install python3-pip
```

## Step 1: Install Python Dependencies

### Option A: Using pip (Recommended)
```bash
# Navigate to project directory
cd /home/sahan/GeneExpressionData-CorrelationMatrixComputing-UsingOpenMP-CUDA

# Install required packages
pip3 install -r benchmarks/requirements.txt

# Or install individually:
pip3 install pandas matplotlib seaborn numpy
```

### Option B: Using apt (Alternative)
```bash
# Install system packages (may be older versions)
sudo apt update
sudo apt install python3-pandas python3-matplotlib python3-seaborn python3-numpy
```

## Step 2: Handle Display Issues in WSL

WSL doesn't have a native display, so we need to configure matplotlib for non-interactive plotting:

### Configure Matplotlib Backend
```bash
# Set matplotlib to use non-interactive backend
export MPLBACKEND=Agg

# Make this permanent by adding to your ~/.bashrc
echo 'export MPLBACKEND=Agg' >> ~/.bashrc
source ~/.bashrc
```

### Alternative: Install X11 Support (Optional)
If you want interactive plots, you can install X11 forwarding:

```bash
# Install X11 packages
sudo apt install x11-apps

# Install VcXsrv on Windows (download from SourceForge)
# Then set DISPLAY variable:
export DISPLAY=:0.0
```

## Step 3: Test the Setup

### Quick Test
```bash
# Test Python packages
python3 -c "import pandas, matplotlib, seaborn, numpy; print('✅ All packages available')"

# Test matplotlib backend
python3 -c "import matplotlib; print('Backend:', matplotlib.get_backend())"
```

### Test Plot Generation
```bash
# Create a simple test plot
cat > test_plot.py << 'EOF'
import matplotlib.pyplot as plt
import numpy as np

# Create simple test plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Test Plot - WSL Environment')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True)

# Save plot (don't show, since we're in WSL)
plt.savefig('test_plot.png', dpi=300, bbox_inches='tight')
print('✅ Test plot saved as test_plot.png')
plt.close()
EOF

# Run the test
python3 test_plot.py

# Check if file was created
ls -la test_plot.png
```

## Step 4: Run Your Benchmarks and Generate Plots

### Complete Workflow

```bash
# 1. Navigate to project directory
cd /home/sahan/GeneExpressionData-CorrelationMatrixComputing-UsingOpenMP-CUDA

# 2. Build all executables if not done already
make clean && make all

# 3. Run quick benchmark
./benchmarks/quick_benchmark.sh

# 4. Generate plots from results
python3 benchmarks/plot_results.py benchmarks/results/quick_benchmark_*.csv
```

### Full Benchmark (Optional - takes longer)
```bash
# Run comprehensive benchmark
./benchmarks/benchmark.sh

# Results will include automatic plot generation
```

## Step 5: View Results

### In WSL
```bash
# List generated files
ls -la benchmarks/results/

# View PNG files (if X11 is set up)
eog benchmarks/results/*.png

# Or copy to Windows to view
cp benchmarks/results/*.png /mnt/c/Users/$USER/Desktop/
```

### In Windows
The plot files will be saved in the WSL filesystem. You can access them at:
```
\\wsl.localhost\Ubuntu\home\sahan\GeneExpressionData-CorrelationMatrixComputing-UsingOpenMP-CUDA\benchmarks\results\
```

Or copy them to your Windows desktop for easy viewing.

## Troubleshooting

### Common Issues and Solutions

#### 1. "No module named 'pandas'"
```bash
# Solution: Install packages
pip3 install pandas matplotlib seaborn numpy
```

#### 2. "No display name and no $DISPLAY environment variable"
```bash
# Solution: Set non-interactive backend
export MPLBACKEND=Agg
```

#### 3. "Permission denied" when installing packages
```bash
# Solution A: Use --user flag
pip3 install --user pandas matplotlib seaborn numpy

# Solution B: Use virtual environment
python3 -m venv plotting_env
source plotting_env/bin/activate
pip3 install pandas matplotlib seaborn numpy
```

#### 4. "Memory Error" when plotting large datasets
```bash
# Solution: Process in chunks or reduce data size
# The plotting script handles this automatically
```

#### 5. Fonts issues in plots
```bash
# Install additional fonts
sudo apt install fonts-liberation fonts-dejavu-core
```

## Expected Output

After successful setup and execution, you should see files like:
```
benchmarks/results/
├── quick_benchmark_20250714_123456.csv
├── execution_times_20250714_123456.png
├── speedup_analysis_20250714_123456.png
├── efficiency_analysis_20250714_123456.png
└── performance_report_20250714_123456.txt
```

## Performance Analysis Features

The plotting script will generate:

1. **Execution Time Analysis**
   - Time vs matrix size plots
   - Distribution box plots
   - Performance heatmaps

2. **Speedup Analysis**
   - Speedup relative to serial baseline
   - Comparative bar charts
   - Scaling trends

3. **Efficiency Analysis**
   - Parallel efficiency metrics
   - Resource utilization
   - Scalability analysis

4. **Comprehensive Reports**
   - Statistical summaries
   - Performance recommendations
   - Best configuration identification

## Next Steps

1. **Run the setup commands above**
2. **Execute the benchmark**
3. **Generate and review plots**
4. **Use insights to optimize your hybrid implementation**

The plots will help you identify:
- Why hybrid performance is lower than expected
- Optimal problem sizes for each implementation
- Bottlenecks in your parallel algorithms
- Scaling behavior across different matrix sizes
