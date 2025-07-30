#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

// CUDA kernel for computing Pearson correlation coefficient
__global__ void cuda_correlation_kernel(float *input_matrix, float *output_matrix, int N, int M)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Only compute upper triangle (including diagonal)
    if (i <= j && i < N && j < N)
    {
        float sum_X = 0.0f, sum_Y = 0.0f, sum_XY = 0.0f;
        float sum_X2 = 0.0f, sum_Y2 = 0.0f;

        // Get pointers to the rows
        float *row_i = input_matrix + i * M;
        float *row_j = input_matrix + j * M;

        // Compute sums for correlation formula
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

        // Store in both positions for symmetric matrix
        output_matrix[i * N + j] = correlation;
        output_matrix[j * N + i] = correlation;
    }
}

// Host function to generate random input matrix
void generate_input_matrix(float *matrix, int n, int m)
{
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

// CUDA correlation computation function
void cuda_correlation(float *h_input_matrix, float *h_output_matrix, int N, int M)
{
    float *d_input_matrix, *d_output_matrix;

    // Calculate memory sizes
    size_t input_size = N * M * sizeof(float);
    size_t output_size = N * N * sizeof(float);

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_input_matrix, input_size));
    CUDA_CHECK(cudaMalloc(&d_output_matrix, output_size));

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(d_input_matrix, h_input_matrix, input_size, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    dim3 block_size(16, 16); // 256 threads per block
    dim3 grid_size((N + block_size.x - 1) / block_size.x, (N + block_size.y - 1) / block_size.y);

    cuda_correlation_kernel<<<grid_size, block_size>>>(d_input_matrix, d_output_matrix, N, M);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output_matrix, d_output_matrix, output_size, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_input_matrix));
    CUDA_CHECK(cudaFree(d_output_matrix));
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

    printf("Executing CUDA Version\n");
    printf("Matrix Size: N=%d, M=%d\n", N, M);

    // Check CUDA device properties
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Allocate host memory
    float *input_matrix = (float *)malloc(N * M * sizeof(float));
    float *output_matrix = (float *)malloc(N * N * sizeof(float));

    if (!input_matrix || !output_matrix)
    {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // Generate input data
    srand(12345);
    generate_input_matrix(input_matrix, N, M);

    print_matrix(input_matrix, N, M, 8, 8, "Input Matrix (Snippet)");

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start time
    CUDA_CHECK(cudaEventRecord(start));

    // .Compute correlation matrix using CUDA
    cuda_correlation(input_matrix, output_matrix, N, M);

    // Record end time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    print_matrix(output_matrix, N, N, 8, 8, "Output Correlation Matrix (Snippet)");
    printf("Execution Time: %f seconds\n", milliseconds / 1000.0f);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    free(input_matrix);
    free(output_matrix);

    return 0;
}
