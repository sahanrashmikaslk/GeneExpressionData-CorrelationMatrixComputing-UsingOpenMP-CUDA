#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

// This is the GPU code that runs in parallel on many threads.
// It computes the correlation for one pair of variables (i, j).
__global__ void correlation_kernel(float* input_matrix, float* output_matrix, int N, int M) {
    // Calculate the unique (row, column) index this thread is responsible for
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundary conditions: make sure we are within the N x N matrix
    // and only compute the upper triangle (where j >= i) to avoid redundant work.
    if (i < N && j < N && j >= i) {
        float sum_X = 0.0f, sum_Y = 0.0f, sum_XY = 0.0f;
        float sum_X2 = 0.0f, sum_Y2 = 0.0f;

        // Pointers to the start of the relevant rows in the input matrix
        float* row_i = input_matrix + i * M;
        float* row_j = input_matrix + j * M;

        // Calculate the sums needed for the Pearson formula
        for (int k = 0; k < M; k++) {
            float val_i = row_i[k];
            float val_j = row_j[k];
            sum_X += val_i;
            sum_Y += val_j;
            sum_XY += val_i * val_j;
            sum_X2 += val_i * val_i;
            sum_Y2 += val_j * val_j;
        }

        // Calculate the final correlation value
        float numerator = (float)M * sum_XY - sum_X * sum_Y;
        float denominator = sqrtf(((float)M * sum_X2 - sum_X * sum_X) * ((float)M * sum_Y2 - sum_Y * sum_Y));
        
        // Store the result, handling division by zero
        output_matrix[i * N + j] = (denominator == 0.0f) ? 1.0f : numerator / denominator;
    }
}

// A simple CPU-side function to fill the lower triangle of the matrix
void fill_symmetric(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            matrix[j * N + i] = matrix[i * N + j];
        }
    }
}

// The main function that runs on the CPU (the "host")
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N_variables> <M_samples>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    printf("Executing Hybrid OpenMP+CUDA Version\n");
    printf("Matrix Size: N=%d, M=%d\n", N, M);
    printf("Num threads available to host: %d\n", omp_get_max_threads());

    // Allocate memory on the CPU (Host)
    float* h_input = (float*)malloc(N * M * sizeof(float));
    float* h_output = (float*)malloc(N * N * sizeof(float));
    if (!h_input || !h_output) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return 1;
    }

    // Initialize input data with random values
    srand(12345);
    for (int i = 0; i < N * M; i++) {
        h_input[i] = (float)rand() / (float)RAND_MAX;
    }

    // Allocate memory on the GPU (Device)
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * M * sizeof(float));
    cudaMalloc(&d_output, N * N * sizeof(float));

    // Copy the input matrix from Host memory to Device memory
    cudaMemcpy(d_input, h_input, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // Define the grid and block dimensions for the CUDA kernel
    dim3 threadsPerBlock(16, 16); // 16x16 = 256 threads per block
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    // Use OpenMP's high-resolution timer to benchmark the GPU computation
    double start_time = omp_get_wtime();

    // Launch the GPU kernel
    correlation_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N, M);
    
    // Block the CPU thread until the GPU kernel is finished
    cudaDeviceSynchronize();

    double end_time = omp_get_wtime();

    // Copy the resulting correlation matrix from Device memory back to Host memory
    cudaMemcpy(h_output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Fill in the symmetric part of the matrix on the CPU
    fill_symmetric(h_output, N);

    printf("Execution Time: %f seconds\n\n", end_time - start_time);

    // Free all allocated memory on both Host and Device
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}