#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>

#define MAX_OUTPUT_SIZE 100000
#define TEMP_DIR "/tmp/corr_verify"

// Structure to hold benchmark results
typedef struct
{
    char implementation[20];
    int N, M;
    double execution_time;
    double rmse;
    int status; // 0 = success, 1 = failed
} BenchmarkResult;

// Function to create temporary directory
void create_temp_dir()
{
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "mkdir -p %s", TEMP_DIR);
    system(cmd);
}

// Function to cleanup temporary files
void cleanup_temp_files()
{
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "rm -rf %s", TEMP_DIR);
    system(cmd);
}

// .. Function to execute a program and capture output
int execute_program(const char *program, int N, int M, float **output_matrix)
{
    char cmd[512];
    char output_file[256];

    snprintf(output_file, sizeof(output_file), "%s/%s_output_%dx%d.txt", TEMP_DIR, program, N, M);
    snprintf(cmd, sizeof(cmd), "./%s %d %d > %s 2>&1", program, N, M, output_file);

    printf("    Executing: %s\n", cmd);
    int ret = system(cmd);

    if (ret != 0)
    {
        printf("      Failed to execute %s\n", program);
        return -1;
    }

    // Parse the output file to extract correlation matrix
    FILE *file = fopen(output_file, "r");
    if (!file)
    {
        printf("      Failed to open output file for %s\n", program);
        return -1;
    }

    char line[1024];
    int reading_matrix = 0;
    int row = 0;
    *output_matrix = (float *)malloc(N * N * sizeof(float));

    while (fgets(line, sizeof(line), file) && row < N)
    {
        if (strstr(line, "--- Output Correlation Matrix"))
        {
            reading_matrix = 1;
            continue;
        }
        if (reading_matrix && strstr(line, "-------"))
        {
            break;
        }
        if (reading_matrix && row < N)
        {
            char *token = strtok(line, " \t\n");
            int col = 0;
            while (token && col < N)
            {
                if (sscanf(token, "%f", &(*output_matrix)[row * N + col]) == 1)
                {
                    col++;
                }
                token = strtok(NULL, " \t\n");
            }
            if (col > 0)
                row++;
        }
    }

    fclose(file);

    // if (row < N)
    // {
    //     printf("      Could not parse complete matrix from %s output\n", program);
    //     free(*output_matrix);
    //     return -1;
    // }

    printf("     Successfully executed %s\n", program);
    return 0;
}

// Function to calculate RMSE between two matrices
double calculate_rmse(float *matrix1, float *matrix2, int N)
{
    double sum_squared_diff = 0.0;
    int count = 0;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double diff = matrix1[i * N + j] - matrix2[i * N + j];
            sum_squared_diff += diff * diff;
            count++;
        }
    }

    return sqrt(sum_squared_diff / count);
}

// Function to extract execution time from output file
double extract_execution_time(const char *program, int N, int M)
{
    char output_file[256];
    snprintf(output_file, sizeof(output_file), "%s/%s_output_%dx%d.txt", TEMP_DIR, program, N, M);

    FILE *file = fopen(output_file, "r");
    if (!file)
        return -1.0;

    char line[1024];
    double exec_time = -1.0;

    while (fgets(line, sizeof(line), file))
    {
        if (strstr(line, "Execution Time:"))
        {
            sscanf(line, "Execution Time: %lf seconds", &exec_time);
            break;
        }
    }

    fclose(file);
    return exec_time;
}

// Function to print results table
void print_results_table(BenchmarkResult *results, int num_results)
{
    printf("\n");
    printf("╔═════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                           ACCURACY VERIFICATION RESULTS                     ║\n");
    printf("╠═════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Implementation │    Matrix Size  │  Exec Time (s)   │     RMSE   │  Status  ║\n");
    printf("╠═════════════════════════════════════════════════════════════════════════════╣\n");

    for (int i = 0; i < num_results; i++)
    {
        char status_symbol[10];
        if (results[i].status == 0)
        {
            if (results[i].rmse < 1e-6)
            {
                strcpy(status_symbol, "PASS");
            }
            else if (results[i].rmse < 1e-3)
            {
                strcpy(status_symbol, "WARN");
            }
            else
            {
                strcpy(status_symbol, "FAIL");
            }
        }
        else
        {
            strcpy(status_symbol, "ERROR");
        }

        printf("║ %-14s │ %6d × %-6d │ %13.6f │ %12.2e │ %-6s ║\n",
               results[i].implementation,
               results[i].N, results[i].M,
               results[i].execution_time,
               results[i].rmse,
               status_symbol);
    }

    printf("╚═══════════════════════════════════════════════════════════════════════════════╝\n");
    // printf("\nAccuracy Criteria:\n");
    // printf(" PASS: RMSE < 1e-6 (Excellent accuracy)\n");
    // printf(" WARN: RMSE < 1e-3 (Acceptable accuracy)\n");
    // printf(" FAIL: RMSE ≥ 1e-3 (Poor accuracy)\n");
    // printf(" ERROR: Execution failed\n\n");
}

// Function to save results to CSV
void save_results_to_csv(BenchmarkResult *results, int num_results)
{
    char filename[256];
    snprintf(filename, sizeof(filename), "benchmarks/results/accuracy_verification_%ld.csv", time(NULL));

    FILE *file = fopen(filename, "w");
    if (!file)
    {
        // printf("Warning: Could not save results to CSV file\n");
        return;
    }

    fprintf(file, "Implementation,Matrix_Size_N,Matrix_Size_M,Execution_Time_s,RMSE,Status,Accuracy_Level\n");

    for (int i = 0; i < num_results; i++)
    {
        char accuracy_level[20];
        if (results[i].status != 0)
        {
            strcpy(accuracy_level, "ERROR");
        }
        else if (results[i].rmse < 1e-6)
        {
            strcpy(accuracy_level, "EXCELLENT");
        }
        else if (results[i].rmse < 1e-3)
        {
            strcpy(accuracy_level, "ACCEPTABLE");
        }
        else
        {
            strcpy(accuracy_level, "POOR");
        }

        fprintf(file, "%s,%d,%d,%.6f,%.2e,%d,%s\n",
                results[i].implementation,
                results[i].N, results[i].M,
                results[i].execution_time,
                results[i].rmse,
                results[i].status,
                accuracy_level);
    }

    fclose(file);
    printf("Results saved to: %s\n", filename);
}

int main(int argc, char **argv)
{

    printf("║                    CORRELATION MATRIX ACCURACY VERIFICATION                  ║\n");


    // Test configurations
    int test_sizes[][2] = {
        {500, 1000},   // Small test
        {1000, 2000},  // Large test
        {2000, 3000}, // Medium test
    };
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    char *implementations[] = {"serial", "openmp", "cuda", "hybrid"};
    int num_implementations = sizeof(implementations) / sizeof(implementations[0]);

    // Check if executables exist
    printf("Checking for required executables...\n");
    for (int i = 0; i < num_implementations; i++)
    {
        if (access(implementations[i], X_OK) != 0)
        {
            printf(" Executable '%s' not found. Please run 'make' first.\n", implementations[i]);
            return 1;
        }
        printf("Found %s\n", implementations[i]);
    }

    create_temp_dir();

    BenchmarkResult *results = malloc(num_tests * num_implementations * sizeof(BenchmarkResult));
    int result_index = 0;

    for (int test = 0; test < num_tests; test++)
    {
        int N = test_sizes[test][0];
        int M = test_sizes[test][1];

        printf("\n Testing matrix size: %d × %d\n", N, M);
        printf("═══════════════════════════════════════\n");

        float *serial_matrix = NULL;
        float *comparison_matrices[num_implementations];
        memset(comparison_matrices, 0, sizeof(comparison_matrices));

        // Execute all implementations
        for (int impl = 0; impl < num_implementations; impl++)
        {
            printf("   Running %s implementation...\n", implementations[impl]);

            BenchmarkResult *current_result = &results[result_index++];
            strcpy(current_result->implementation, implementations[impl]);
            current_result->N = N;
            current_result->M = M;
            current_result->rmse = 0.0;
            current_result->status = 0;

            if (execute_program(implementations[impl], N, M, &comparison_matrices[impl]) != 0)
            {
                current_result->status = 1;
                current_result->execution_time = -1.0;
                continue;
            }

            current_result->execution_time = extract_execution_time(implementations[impl], N, M);

            // Use serial as reference
            if (strcmp(implementations[impl], "serial") == 0)
            {
                serial_matrix = comparison_matrices[impl];
                current_result->rmse = 0.0; // Serial vs itself
            }
            else if (serial_matrix != NULL)
            {
                current_result->rmse = calculate_rmse(serial_matrix, comparison_matrices[impl], N);
                printf("     RMSE vs Serial: %.2e\n", current_result->rmse);
            }
        }

        // Cleanup matrices
        for (int impl = 0; impl < num_implementations; impl++)
        {
            if (comparison_matrices[impl])
            {
                free(comparison_matrices[impl]);
            }
        }
    }

    // Print results table
    print_results_table(results, result_index);

    // Save results to CSV
    save_results_to_csv(results, result_index);

    // Summary statistics
    int passed = 0, warned = 0, failed = 0, errors = 0;
    for (int i = 0; i < result_index; i++)
    {
        if (results[i].status != 0)
        {
            errors++;
        }
        else if (strcmp(results[i].implementation, "serial") == 0)
        {
            // Skip serial in summary
        }
        else if (results[i].rmse < 1e-6)
        {
            passed++;
        }
        else if (results[i].rmse < 1e-3)
        {
            warned++;
        }
        else
        {
            failed++;
        }
    }

    // printf(" SUMMARY:\n");
    // printf("    Excellent accuracy: %d implementations\n", passed);
    // printf("     Acceptable accuracy: %d implementations\n", warned);
    // printf("     Poor accuracy: %d implementations\n", failed);
    // printf("     Execution errors: %d implementations\n", errors);

    cleanup_temp_files();
    free(results);

  
    return 0;
}
