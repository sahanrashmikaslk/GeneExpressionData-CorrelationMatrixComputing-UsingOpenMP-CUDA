
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h> // For access()

#define TEMP_DIR "/tmp/corr_benchmark"

// Structure to hold benchmark results
typedef struct
{
    char implementation[20];
    int N, M;
    int threads; // To store the number of OpenMP threads used
    double execution_time;
    char status[20]; // e.g., "SUCCESS", "FAILED"
} BenchmarkResult;

// --- Benchmark Configuration ---
int test_sizes[][2] = {
    {1000, 2000}, // Medium
    {2000, 4000}  // Large
};
char *implementations[] = {"serial", "openmp", "cuda", "hybrid"};
// Define the thread counts to test for OpenMP and Hybrid
int omp_thread_counts[] = {1, 2, 4, 8, 12, 16};
// --- End Configuration ---

// Function to create a temporary directory for log files
void create_temp_dir()
{
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", TEMP_DIR);
    system(cmd);
}

// Function to clean up the temporary directory
void cleanup_temp_dir()
{
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", TEMP_DIR);
    system(cmd);
}

// Function to execute a program and get its status
int execute_program(const char *program, int N, int M, int threads)
{
    char cmd[512];
    char output_file[256];
    char thread_str[4];

    // Set the OMP_NUM_THREADS environment variable for this specific execution
    // This is the key to testing different thread counts.
    if (threads > 0) {
        snprintf(thread_str, sizeof(thread_str), "%d", threads);
        setenv("OMP_NUM_THREADS", thread_str, 1); // 1 = overwrite existing value
    }

    snprintf(output_file, sizeof(output_file), "%s/%s_%dx%d_T%d.log", TEMP_DIR, program, N, M, threads);
    // Execute the program, redirecting all output to the log file
    snprintf(cmd, sizeof(cmd), "./%s %d %d > %s 2>&1", program, N, M, output_file);

    printf("      Executing (T=%d): ./%s %d %d\n", threads, program, N, M);
    return system(cmd);
}

// Function to extract execution time from the program's log file
double extract_execution_time(const char *program, int N, int M, int threads)
{
    char output_file[256];
    snprintf(output_file, sizeof(output_file), "%s/%s_%dx%d_T%d.log", TEMP_DIR, program, N, M, threads);

    FILE *file = fopen(output_file, "r");
    if (!file)
        return -1.0;

    char line[1024];
    double exec_time = -1.0;

    while (fgets(line, sizeof(line), file))
    {
        // Look for the "Execution Time:" line printed by the programs
        if (sscanf(line, "Execution Time: %lf seconds", &exec_time) == 1)
        {
            break;
        }
    }

    fclose(file);
    return exec_time;
}

// Function to print the final results in a formatted table
void print_results_table(BenchmarkResult *results, int num_results)
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                       PERFORMANCE BENCHMARK RESULTS                       ║\n");
    printf("╠══════════════════╦═════════════════╦═════════╦═══════════════════╦═════════╣\n");
    printf("║ Implementation   │ Matrix Size     │ Threads │ Exec Time (s)     │ Status  ║\n");
    printf("╠══════════════════╬═════════════════╬═════════╬═══════════════════╬═════════╣\n");

    for (int i = 0; i < num_results; i++)
    {
        char size_str[20];
        snprintf(size_str, sizeof(size_str), "%d x %d", results[i].N, results[i].M);

        // Display threads as "N/A" for non-OpenMP versions for clarity
        char thread_str[10];
        if (results[i].threads == 0) {
            snprintf(thread_str, sizeof(thread_str), "N/A");
        } else {
            snprintf(thread_str, sizeof(thread_str), "%d", results[i].threads);
        }
        
        printf("║ %-16s │ %-15s │ %-7s │ %-17.6f │ %-7s ║\n",
               results[i].implementation,
               size_str,
               thread_str,
               results[i].execution_time,
               results[i].status);
    }

    printf("╚══════════════════╩═════════════════╩═════════╩═══════════════════╩═════════╝\n");
}

// Function to save results to a CSV file for plotting
void save_results_to_csv(BenchmarkResult *results, int num_results)
{
    char filename[256];
    snprintf(filename, sizeof(filename), "benchmarks/results/performance_results_%ld.csv", time(NULL));

    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("\nWarning: Could not save results to CSV file '%s'\n", filename);
        return;
    }

    fprintf(file, "Implementation,N,M,Threads,ExecutionTime_s,Status\n");

    for (int i = 0; i < num_results; i++) {
        fprintf(file, "%s,%d,%d,%d,%.6f,%s\n",
                results[i].implementation,
                results[i].N, results[i].M,
                results[i].threads,
                results[i].execution_time,
                results[i].status);
    }

    fclose(file);
    printf("\nBenchmark results saved to: %s\n", filename);
}

int main(void)
{
    printf("Starting Performance Benchmark Suite...\n");

    int num_test_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);
    int num_implementations = sizeof(implementations) / sizeof(implementations[0]);
    int num_omp_threads = sizeof(omp_thread_counts) / sizeof(omp_thread_counts[0]);

    // Check if executables exist before starting
    printf("Checking for required executables...\n");
    for (int i = 0; i < num_implementations; i++) {
        if (access(implementations[i], X_OK) != 0) {
            fprintf(stderr, "Error: Executable '%s' not found or not executable. Please run 'make' in the project root.\n", implementations[i]);
            return 1;
        }
    }
    printf("All executables found.\n\n");

    create_temp_dir();

    // Allocate space for all possible results
    int max_results = num_test_sizes * (2 + 2 * num_omp_threads); // 2 for serial/cuda, 2*omp for openmp/hybrid
    BenchmarkResult *results = malloc(max_results * sizeof(BenchmarkResult));
    int result_index = 0;

    for (int i = 0; i < num_test_sizes; i++) {
        int N = test_sizes[i][0];
        int M = test_sizes[i][1];

        printf("--- Testing Matrix Size: %d x %d ---\n", N, M);

        for (int j = 0; j < num_implementations; j++) {
            const char *impl = implementations[j];
            
            // For openmp and hybrid, loop through all specified thread counts
            if (strcmp(impl, "openmp") == 0 || strcmp(impl, "hybrid") == 0) {
                printf("  Benchmarking '%s' with multiple threads...\n", impl);
                for (int t = 0; t < num_omp_threads; t++) {
                    int threads = omp_thread_counts[t];
                    BenchmarkResult *res = &results[result_index++];
                    
                    strcpy(res->implementation, impl);
                    res->N = N;
                    res->M = M;
                    res->threads = threads;

                    if (execute_program(impl, N, M, threads) == 0) {
                        res->execution_time = extract_execution_time(impl, N, M, threads);
                        strcpy(res->status, "SUCCESS");
                    } else {
                        res->execution_time = -1.0;
                        strcpy(res->status, "FAILED");
                    }
                }
            } else { // For serial and cuda, run only once
                printf("  Benchmarking '%s'...\n", impl);
                BenchmarkResult *res = &results[result_index++];

                strcpy(res->implementation, impl);
                res->N = N;
                res->M = M;
                res->threads = (strcmp(impl, "serial") == 0) ? 1 : 0; // 1 for serial, 0 (N/A) for CUDA

                if (execute_program(impl, N, M, res->threads) == 0) {
                    res->execution_time = extract_execution_time(impl, N, M, res->threads);
                    strcpy(res->status, "SUCCESS");
                } else {
                    res->execution_time = -1.0;
                    strcpy(res->status, "FAILED");
                }
            }
        }
        printf("-------------------------------------------\n\n");
    }

    // Print and save results
    print_results_table(results, result_index);
    save_results_to_csv(results, result_index);

    // Cleanup
    cleanup_temp_dir();
    free(results);

    return 0;
}