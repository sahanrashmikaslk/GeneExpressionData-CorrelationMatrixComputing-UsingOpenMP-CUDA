#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CUDA_CHECK(call)                                                                                  \
    do                                                                                                    \
    {                                                                                                     \
        cudaError_t error = call;                                                                         \
        if (error != cudaSuccess)                                                                         \
        {                                                                                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE);                                                                           \
        }                                                                                                 \
    } while (0)

// CUDA kernel for computing correlation coefficients
__global__ void hybrid_correlation_kernel(float *input_matrix, float *output_matrix, int N, int M, int start_row, int end_row)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y + start_row;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Only compute upper triangle within the assigned row range
    if (i <= j && i < end_row && i >= start_row && j < N)
    {
        float sum_X = 0.0f, sum_Y = 0.0f, sum_XY = 0.0f;
        float sum_X2 = 0.0f, sum_Y2 = 0.0f;

        float *row_i = input_matrix + i * M;
        float *row_j = input_matrix + j * M;

        // Compute correlation statistics
        for (int k = 0; k < M; k++)
        {
            float xi = row_i[k];
            float xj = row_j[k];

            sum_X += xi;
            sum_Y += xj;
            sum_XY += xi * xj;
            sum_X2 += xi * xi;
            sum_Y2 += xj * xj;
        }

        // Calculate Pearson correlation coefficient
        float numerator = (float)M * sum_XY - sum_X * sum_Y;
        float denominator = sqrtf(((float)M * sum_X2 - sum_X * sum_X) * ((float)M * sum_Y2 - sum_Y * sum_Y));

        float correlation = (denominator == 0.0f) ? 1.0f : numerator / denominator;

        // Store correlation value
        output_matrix[i * N + j] = correlation;
        if (i != j)
        {
            output_matrix[j * N + i] = correlation;
        }
    }
}

// Host function to generate random input matrix
void generate_input_matrix(float *matrix, int n, int m)
{
#pragma omp parallel for
    for (int i = 0; i < n * m; i++)
    {
        matrix[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Host function to print matrix
void print_matrix(const float *matrix, int total_rows, int total_cols, int rows_to_print, int cols_to_print, const char *title)
{
    printf("--- %s ---\n", title);

    if (rows_to_print > total_rows)
        rows_to_print = total_rows;
    if (cols_to_print > total_cols)
        cols_to_print = total_cols;

    for (int i = 0; i < rows_to_print; i++)
    {
        for (int j = 0; j < cols_to_print; j++)
        {
            printf("%8.4f ", matrix[i * total_cols + j]);
        }
        printf("\n");
    }
    printf("---------------------------------------\n\n");
}

// Hybrid correlation computation using OpenMP + CUDA
void hybrid_correlation(float *h_input_matrix, float *h_output_matrix, int N, int M)
{
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

    if (num_gpus == 0)
    {
        fprintf(stderr, "No CUDA devices available!\n");
        return;
    }

    printf("Using %d GPU(s) with OpenMP threads\n", num_gpus);

// Initialize output matrix
#pragma omp parallel for
    for (int i = 0; i < N * N; i++)
    {
        h_output_matrix[i] = 0.0f;
    }

    // Determine optimal work distribution
    int rows_per_gpu = N / num_gpus;
    int remaining_rows = N % num_gpus;

// Use OpenMP to manage multiple GPUs
#pragma omp parallel num_threads(num_gpus)
    {
        int thread_id = omp_get_thread_num();
        int gpu_id = thread_id % num_gpus;

        // Set device for this thread
        CUDA_CHECK(cudaSetDevice(gpu_id));

        // Calculate row range for this GPU
        int start_row = thread_id * rows_per_gpu;
        int end_row = start_row + rows_per_gpu;
        if (thread_id == num_gpus - 1)
        {
            end_row += remaining_rows;
        }

        int local_rows = end_row - start_row;

        if (local_rows > 0)
        {
            // Allocate GPU memory
            float *d_input_matrix, *d_output_matrix;
            size_t input_size = N * M * sizeof(float);
            size_t output_size = N * N * sizeof(float);

            CUDA_CHECK(cudaMalloc(&d_input_matrix, input_size));
            CUDA_CHECK(cudaMalloc(&d_output_matrix, output_size));

            // Copy input data to GPU
            CUDA_CHECK(cudaMemcpy(d_input_matrix, h_input_matrix, input_size, cudaMemcpyHostToDevice));

            // Initialize output matrix on GPU
            CUDA_CHECK(cudaMemset(d_output_matrix, 0, output_size));

            // Configure kernel launch parameters
            dim3 block_size(16, 16);
            dim3 grid_size((N + block_size.x - 1) / block_size.x,
                           (local_rows + block_size.y - 1) / block_size.y);

            // Launch kernel for this GPU's portion
            hybrid_correlation_kernel<<<grid_size, block_size>>>(
                d_input_matrix, d_output_matrix, N, M, start_row, end_row);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Copy partial results back to host
            float *temp_output = (float *)malloc(output_size);
            CUDA_CHECK(cudaMemcpy(temp_output, d_output_matrix, output_size, cudaMemcpyDeviceToHost));

// Merge results into main output matrix (thread-safe)
#pragma omp critical
            {
                for (int i = start_row; i < end_row; i++)
                {
                    for (int j = i; j < N; j++)
                    {
                        h_output_matrix[i * N + j] = temp_output[i * N + j];
                        h_output_matrix[j * N + i] = temp_output[i * N + j];
                    }
                }
            }

            // Cleanup
            free(temp_output);
            CUDA_CHECK(cudaFree(d_input_matrix));
            CUDA_CHECK(cudaFree(d_output_matrix));
        }
    }
}

// Alternative hybrid approach: CPU preprocessing + GPU computation
void hybrid_correlation_alternative(float *h_input_matrix, float *h_output_matrix, int N, int M)
{
    printf("Using alternative hybrid approach: CPU preprocessing + GPU computation\n");

    // Step 1: Use OpenMP to precompute row statistics on CPU
    float *row_means = (float *)malloc(N * sizeof(float));
    float *row_vars = (float *)malloc(N * sizeof(float));

#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int j = 0; j < M; j++)
        {
            float val = h_input_matrix[i * M + j];
            sum += val;
            sum_sq += val * val;
        }

        row_means[i] = sum / M;
        row_vars[i] = (sum_sq / M) - (row_means[i] * row_means[i]);
    }

    // Step 2: Use CUDA for correlation computation with preprocessed data
    float *d_input_matrix, *d_output_matrix, *d_row_means, *d_row_vars;

    size_t input_size = N * M * sizeof(float);
    size_t output_size = N * N * sizeof(float);
    size_t stats_size = N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_input_matrix, input_size));
    CUDA_CHECK(cudaMalloc(&d_output_matrix, output_size));
    CUDA_CHECK(cudaMalloc(&d_row_means, stats_size));
    CUDA_CHECK(cudaMalloc(&d_row_vars, stats_size));

    CUDA_CHECK(cudaMemcpy(d_input_matrix, h_input_matrix, input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_means, row_means, stats_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_vars, row_vars, stats_size, cudaMemcpyHostToDevice));

    // Launch optimized kernel
    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    hybrid_correlation_kernel<<<grid_size, block_size>>>(d_input_matrix, d_output_matrix, N, M, 0, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output_matrix, d_output_matrix, output_size, cudaMemcpyDeviceToHost));

    // Cleanup
    free(row_means);
    free(row_vars);
    CUDA_CHECK(cudaFree(d_input_matrix));
    CUDA_CHECK(cudaFree(d_output_matrix));
    CUDA_CHECK(cudaFree(d_row_means));
    CUDA_CHECK(cudaFree(d_row_vars));
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <N_variables> <M_samples>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    printf("Executing Hybrid (OpenMP + CUDA) Version\n");
    printf("Matrix Size: N=%d, M=%d\n", N, M);
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    // Check CUDA devices
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }

    for (int i = 0; i < device_count; i++)
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("GPU %d: %s (Compute %d.%d, %.2f GB)\n", i, prop.name,
               prop.major, prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }

    // Allocate host memory
    float *input_matrix = (float *)malloc(N * M * sizeof(float));
    float *output_matrix = (float *)malloc(N * N * sizeof(float));

    if (!input_matrix || !output_matrix)
    {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // Generate input data using OpenMP
    srand(12345);
    generate_input_matrix(input_matrix, N, M);

    print_matrix(input_matrix, N, M, 8, 8, "Input Matrix (Snippet)");

    // Time the hybrid computation
    double start_time = omp_get_wtime();

    // Choose between two hybrid approaches based on problem size
    if (N > 1000 && device_count > 1)
    {
        hybrid_correlation(input_matrix, output_matrix, N, M);
    }
    else
    {
        hybrid_correlation_alternative(input_matrix, output_matrix, N, M);
    }

    double end_time = omp_get_wtime();

    print_matrix(output_matrix, N, N, 8, 8, "Output Correlation Matrix (Snippet)");
    printf("Execution Time: %f seconds\n", end_time - start_time);

    free(input_matrix);
    free(output_matrix);

    return 0;
}
