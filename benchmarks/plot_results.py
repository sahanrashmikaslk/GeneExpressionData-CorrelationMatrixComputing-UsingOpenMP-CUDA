#!/usr/bin/env python3
"""
Performance Analysis and Plotting Script for Correlation Matrix Computing
Author: Lelwala L.G.S.R (EG/2020/4047)
Project: Large Pearson Correlation Matrix Computing Using Hybrid OpenMP and CUDA

This script generates comprehensive performance plots and analysis from benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os
from pathlib import Path
import argparse
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PerformanceAnalyzer:
    def __init__(self, csv_file):
        """Initialize the performance analyzer with benchmark data."""
        self.csv_file = csv_file
        self.data = None
        self.output_dir = Path(csv_file).parent
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_data(self):
        """Load and preprocess the benchmark data."""
        try:
            self.data = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.data)} benchmark records")
            
            # Filter only successful runs
            self.data = self.data[self.data['Status'] == 'SUCCESS'].copy()
            print(f"Found {len(self.data)} successful benchmark records")
            
            # Convert execution time to numeric, handle errors
            self.data['ExecutionTime(s)'] = pd.to_numeric(self.data['ExecutionTime(s)'], errors='coerce')
            self.data = self.data.dropna(subset=['ExecutionTime(s)'])
            
            # Calculate speedup relative to serial implementation
            self.calculate_speedup()
            
            # Calculate efficiency metrics
            self.calculate_efficiency()
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def calculate_speedup(self):
        """Calculate speedup relative to serial implementation."""
        speedup_data = []
        
        for n in self.data['N'].unique():
            for m in self.data['M'].unique():
                # Get serial baseline
                serial_times = self.data[
                    (self.data['Implementation'] == 'serial') & 
                    (self.data['N'] == n) & 
                    (self.data['M'] == m)
                ]['ExecutionTime(s)']
                
                if len(serial_times) > 0:
                    serial_avg = serial_times.mean()
                    
                    # Calculate speedup for each implementation
                    for impl in self.data['Implementation'].unique():
                        impl_times = self.data[
                            (self.data['Implementation'] == impl) & 
                            (self.data['N'] == n) & 
                            (self.data['M'] == m)
                        ]['ExecutionTime(s)']
                        
                        if len(impl_times) > 0:
                            impl_avg = impl_times.mean()
                            speedup = serial_avg / impl_avg
                            
                            speedup_data.append({
                                'N': n, 'M': m, 'Implementation': impl,
                                'Speedup': speedup, 'SerialTime': serial_avg,
                                'ImplTime': impl_avg
                            })
        
        self.speedup_df = pd.DataFrame(speedup_data)
    
    def calculate_efficiency(self):
        """Calculate parallel efficiency metrics."""
        # This is a simplified efficiency calculation
        # In reality, you'd need to know the exact number of cores/threads used
        self.efficiency_df = self.speedup_df.copy()
        
        # Assume max efficiency baseline (you can adjust these based on your system)
        max_threads = 8  # Adjust based on your CPU cores
        max_cuda_cores = 384  # Adjust based on your GPU (MX230 has 384 cores)
        
        def calculate_efficiency(row):
            if row['Implementation'] == 'serial':
                return 1.0
            elif row['Implementation'] == 'openmp':
                return row['Speedup'] / max_threads
            elif row['Implementation'] == 'cuda':
                return row['Speedup'] / (max_cuda_cores / 32)  # Normalized
            elif row['Implementation'] == 'hybrid':
                return row['Speedup'] / (max_threads + max_cuda_cores / 32)  # Combined
            return 0.0
        
        self.efficiency_df['Efficiency'] = self.efficiency_df.apply(calculate_efficiency, axis=1)
    
    def plot_execution_times(self):
        """Plot execution times for different implementations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Execution Time Analysis by Implementation', fontsize=16, fontweight='bold')
        
        # Average execution times by matrix size
        avg_times = self.data.groupby(['Implementation', 'N', 'M'])['ExecutionTime(s)'].mean().reset_index()
        
        # Plot 1: Execution time vs N (for different M values)
        ax1 = axes[0, 0]
        for m in sorted(avg_times['M'].unique())[:3]:  # Show only first 3 M values
            subset = avg_times[avg_times['M'] == m]
            for impl in subset['Implementation'].unique():
                impl_data = subset[subset['Implementation'] == impl]
                ax1.plot(impl_data['N'], impl_data['ExecutionTime(s)'], 
                        marker='o', label=f'{impl} (M={m})', linewidth=2)
        
        ax1.set_xlabel('Number of Variables (N)')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Execution Time vs Number of Variables')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Execution time vs M (for different N values)
        ax2 = axes[0, 1]
        for n in sorted(avg_times['N'].unique())[:3]:  # Show only first 3 N values
            subset = avg_times[avg_times['N'] == n]
            for impl in subset['Implementation'].unique():
                impl_data = subset[subset['Implementation'] == impl]
                ax2.plot(impl_data['M'], impl_data['ExecutionTime(s)'], 
                        marker='s', label=f'{impl} (N={n})', linewidth=2)
        
        ax2.set_xlabel('Number of Samples (M)')
        ax2.set_ylabel('Execution Time (seconds)')
        ax2.set_title('Execution Time vs Number of Samples')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Box plot of execution times by implementation
        ax3 = axes[1, 0]
        self.data.boxplot(column='ExecutionTime(s)', by='Implementation', ax=ax3)
        ax3.set_title('Execution Time Distribution by Implementation')
        ax3.set_xlabel('Implementation')
        ax3.set_ylabel('Execution Time (seconds)')
        ax3.set_yscale('log')
        
        # Plot 4: Heatmap of average execution times
        ax4 = axes[1, 1]
        
        # Create pivot table for heatmap
        heatmap_data = avg_times.groupby(['Implementation', 'N'])['ExecutionTime(s)'].mean().unstack()
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax4)
        ax4.set_title('Average Execution Time Heatmap\n(Implementation vs N)')
        ax4.set_xlabel('Number of Variables (N)')
        ax4.set_ylabel('Implementation')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f'execution_times_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Execution time plots saved to: {output_file}")
    
    def plot_speedup_analysis(self):
        """Plot speedup analysis."""
        if not hasattr(self, 'speedup_df'):
            print("Speedup data not available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Speedup Analysis Relative to Serial Implementation', fontsize=16, fontweight='bold')
        
        # Plot 1: Speedup vs N
        ax1 = axes[0, 0]
        for impl in self.speedup_df['Implementation'].unique():
            if impl != 'serial':
                impl_data = self.speedup_df[self.speedup_df['Implementation'] == impl]
                avg_speedup = impl_data.groupby('N')['Speedup'].mean()
                ax1.plot(avg_speedup.index, avg_speedup.values, 
                        marker='o', label=impl, linewidth=2, markersize=8)
        
        ax1.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Serial Baseline')
        ax1.set_xlabel('Number of Variables (N)')
        ax1.set_ylabel('Speedup Factor')
        ax1.set_title('Average Speedup vs Number of Variables')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup vs M
        ax2 = axes[0, 1]
        for impl in self.speedup_df['Implementation'].unique():
            if impl != 'serial':
                impl_data = self.speedup_df[self.speedup_df['Implementation'] == impl]
                avg_speedup = impl_data.groupby('M')['Speedup'].mean()
                ax2.plot(avg_speedup.index, avg_speedup.values, 
                        marker='s', label=impl, linewidth=2, markersize=8)
        
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Serial Baseline')
        ax2.set_xlabel('Number of Samples (M)')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Average Speedup vs Number of Samples')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Speedup comparison bar chart
        ax3 = axes[1, 0]
        speedup_summary = self.speedup_df[self.speedup_df['Implementation'] != 'serial'].groupby('Implementation')['Speedup'].agg(['mean', 'std']).reset_index()
        
        bars = ax3.bar(speedup_summary['Implementation'], speedup_summary['mean'], 
                      yerr=speedup_summary['std'], capsize=5, alpha=0.8)
        ax3.set_xlabel('Implementation')
        ax3.set_ylabel('Average Speedup Factor')
        ax3.set_title('Overall Average Speedup with Standard Deviation')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, speedup_summary['mean']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{mean_val:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Speedup heatmap
        ax4 = axes[1, 1]
        speedup_pivot = self.speedup_df.pivot_table(values='Speedup', 
                                                   index='Implementation', 
                                                   columns='N', 
                                                   aggfunc='mean')
        
        sns.heatmap(speedup_pivot, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax4, 
                   center=1, vmin=0, vmax=speedup_pivot.max().max())
        ax4.set_title('Speedup Heatmap (Implementation vs N)')
        ax4.set_xlabel('Number of Variables (N)')
        ax4.set_ylabel('Implementation')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f'speedup_analysis_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Speedup analysis plots saved to: {output_file}")
    
    def plot_efficiency_analysis(self):
        """Plot parallel efficiency analysis."""
        if not hasattr(self, 'efficiency_df'):
            print("Efficiency data not available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parallel Efficiency Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Efficiency vs problem size
        ax1 = axes[0, 0]
        for impl in self.efficiency_df['Implementation'].unique():
            if impl != 'serial':
                impl_data = self.efficiency_df[self.efficiency_df['Implementation'] == impl]
                # Use total problem size (N*M) as x-axis
                impl_data = impl_data.copy()
                impl_data['ProblemSize'] = impl_data['N'] * impl_data['M']
                avg_efficiency = impl_data.groupby('ProblemSize')['Efficiency'].mean()
                ax1.plot(avg_efficiency.index, avg_efficiency.values, 
                        marker='o', label=impl, linewidth=2)
        
        ax1.set_xlabel('Problem Size (N × M)')
        ax1.set_ylabel('Efficiency')
        ax1.set_title('Parallel Efficiency vs Problem Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        
        # Plot 2: Efficiency distribution
        ax2 = axes[0, 1]
        efficiency_data = self.efficiency_df[self.efficiency_df['Implementation'] != 'serial']
        efficiency_data.boxplot(column='Efficiency', by='Implementation', ax=ax2)
        ax2.set_title('Efficiency Distribution by Implementation')
        ax2.set_xlabel('Implementation')
        ax2.set_ylabel('Efficiency')
        
        # Plot 3: Scalability analysis
        ax3 = axes[1, 0]
        for impl in ['openmp', 'cuda', 'hybrid']:
            if impl in self.efficiency_df['Implementation'].values:
                impl_data = self.efficiency_df[self.efficiency_df['Implementation'] == impl]
                avg_speedup = impl_data.groupby('N')['Speedup'].mean()
                ax3.plot(avg_speedup.index, avg_speedup.values, 
                        marker='o', label=f'{impl} Speedup', linewidth=2)
        
        # Add ideal speedup lines for reference
        n_values = sorted(self.efficiency_df['N'].unique())
        ax3.plot(n_values, [8] * len(n_values), '--', alpha=0.5, label='Ideal OpenMP (8 cores)')
        ax3.plot(n_values, [12] * len(n_values), '--', alpha=0.5, label='GPU Baseline')
        
        ax3.set_xlabel('Number of Variables (N)')
        ax3.set_ylabel('Speedup Factor')
        ax3.set_title('Scalability Analysis')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Implementation comparison
        ax4 = axes[1, 1]
        
        # Calculate average metrics for each implementation
        summary_stats = []
        for impl in self.efficiency_df['Implementation'].unique():
            if impl != 'serial':
                impl_data = self.efficiency_df[self.efficiency_df['Implementation'] == impl]
                summary_stats.append({
                    'Implementation': impl,
                    'Avg_Speedup': impl_data['Speedup'].mean(),
                    'Avg_Efficiency': impl_data['Efficiency'].mean(),
                    'Std_Speedup': impl_data['Speedup'].std()
                })
        
        summary_df = pd.DataFrame(summary_stats)
        
        x = np.arange(len(summary_df))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, summary_df['Avg_Speedup'], width, 
                       label='Average Speedup', alpha=0.8)
        bars2 = ax4.bar(x + width/2, summary_df['Avg_Efficiency'] * 10, width, 
                       label='Average Efficiency (×10)', alpha=0.8)
        
        ax4.set_xlabel('Implementation')
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Summary Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(summary_df['Implementation'])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = self.output_dir / f'efficiency_analysis_{self.timestamp}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Efficiency analysis plots saved to: {output_file}")
    
    def generate_performance_report(self):
        """Generate a comprehensive performance report."""
        report_file = self.output_dir / f'performance_report_{self.timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PERFORMANCE ANALYSIS REPORT\n")
            f.write("Correlation Matrix Computing - Hybrid OpenMP + CUDA\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data source: {self.csv_file}\n\n")
            
            # Basic statistics
            f.write("BASIC STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total benchmark runs: {len(self.data)}\n")
            f.write(f"Implementations tested: {', '.join(self.data['Implementation'].unique())}\n")
            f.write(f"Matrix sizes (N): {sorted(self.data['N'].unique())}\n")
            f.write(f"Sample sizes (M): {sorted(self.data['M'].unique())}\n\n")
            
            # Execution time statistics
            f.write("EXECUTION TIME STATISTICS (seconds)\n")
            f.write("-" * 40 + "\n")
            exec_stats = self.data.groupby('Implementation')['ExecutionTime(s)'].agg(['count', 'mean', 'std', 'min', 'max'])
            f.write(exec_stats.to_string())
            f.write("\n\n")
            
            # Speedup statistics
            if hasattr(self, 'speedup_df'):
                f.write("SPEEDUP STATISTICS\n")
                f.write("-" * 40 + "\n")
                speedup_stats = self.speedup_df[self.speedup_df['Implementation'] != 'serial'].groupby('Implementation')['Speedup'].agg(['count', 'mean', 'std', 'min', 'max'])
                f.write(speedup_stats.to_string())
                f.write("\n\n")
                
                # Best performance cases
                f.write("BEST PERFORMANCE CASES\n")
                f.write("-" * 40 + "\n")
                best_speedup = self.speedup_df.loc[self.speedup_df['Speedup'].idxmax()]
                f.write(f"Highest speedup: {best_speedup['Speedup']:.2f}x\n")
                f.write(f"  Implementation: {best_speedup['Implementation']}\n")
                f.write(f"  Matrix size: N={best_speedup['N']}, M={best_speedup['M']}\n\n")
            
            # Performance recommendations
            f.write("PERFORMANCE RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            if hasattr(self, 'speedup_df'):
                avg_speedups = self.speedup_df[self.speedup_df['Implementation'] != 'serial'].groupby('Implementation')['Speedup'].mean().sort_values(ascending=False)
                
                f.write("Implementation ranking by average speedup:\n")
                for i, (impl, speedup) in enumerate(avg_speedups.items(), 1):
                    f.write(f"  {i}. {impl}: {speedup:.2f}x speedup\n")
                
                f.write("\nRecommendations:\n")
                best_impl = avg_speedups.index[0]
                f.write(f"• For overall best performance, use: {best_impl}\n")
                
                if 'hybrid' in avg_speedups.index:
                    hybrid_speedup = avg_speedups['hybrid']
                    if 'cuda' in avg_speedups.index:
                        cuda_speedup = avg_speedups['cuda']
                        if hybrid_speedup > cuda_speedup:
                            f.write("• Hybrid approach successfully outperforms pure CUDA\n")
                        else:
                            f.write("• Hybrid approach needs optimization - currently underperforming vs pure CUDA\n")
                
                f.write("• Consider problem size when choosing implementation\n")
                f.write("• Monitor GPU memory usage for large matrices\n")
            
        print(f"Performance report saved to: {report_file}")
    
    def run_full_analysis(self):
        """Run the complete performance analysis."""
        print("Starting performance analysis...")
        
        if not self.load_data():
            return False
        
        print("Generating execution time plots...")
        self.plot_execution_times()
        
        print("Generating speedup analysis...")
        self.plot_speedup_analysis()
        
        print("Generating efficiency analysis...")
        self.plot_efficiency_analysis()
        
        print("Generating performance report...")
        self.generate_performance_report()
        
        print("Performance analysis completed!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Analyze and plot performance results')
    parser.add_argument('csv_file', help='Path to the benchmark results CSV file')
    parser.add_argument('--output-dir', help='Output directory for plots (default: same as CSV file)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file '{args.csv_file}' not found")
        sys.exit(1)
    
    analyzer = PerformanceAnalyzer(args.csv_file)
    
    if args.output_dir:
        analyzer.output_dir = Path(args.output_dir)
        analyzer.output_dir.mkdir(exist_ok=True)
    
    success = analyzer.run_full_analysis()
    
    if success:
        print(f"\nAll analysis files saved to: {analyzer.output_dir}")
        sys.exit(0)
    else:
        print("Analysis failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
