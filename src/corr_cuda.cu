#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// CUDA Kernel to compute the correlation matrix
__global__ void correlation_kernel(float* input_matrix, float* output_matrix, int N, int M) {
    // Calculate the global thread ID for the output matrix
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to ensure we don't go out of bounds and compute only the upper triangle
    if (i < N && j < N && j >= i) {
        float sum_X = 0.0, sum_Y = 0.0, sum_XY = 0.0;
        float sum_X2 = 0.0, sum_Y2 = 0.0;

        float* row_i = input_matrix + i * M;
        float* row_j = input_matrix + j * M;

        for (int k = 0; k < M; k++) {
            float val_i = row_i[k];
            float val_j = row_j[k];
            sum_X += val_i;
            sum_Y += val_j;
            sum_XY += val_i * val_j;
            sum_X2 += val_i * val_i;
            sum_Y2 += val_j * val_j;
        }

        float numerator = M * sum_XY - sum_X * sum_Y;
        float denominator = sqrtf((M * sum_X2 - sum_X * sum_X) * (M * sum_Y2 - sum_Y * sum_Y));
        
        output_matrix[i * N + j] = (denominator == 0) ? 1.0f : numerator / denominator;
    }
}

// Helper to fill the symmetric part of the matrix on the CPU
void fill_symmetric(float* matrix, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            matrix[j * N + i] = matrix[i * N + j];
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N_variables> <M_samples>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    printf("Executing CUDA Version\n");
    printf("Matrix Size: N=%d, M=%d\n", N, M);

    // Host memory allocation
    float* h_input = (float*)malloc(N * M * sizeof(float));
    float* h_output = (float*)malloc(N * N * sizeof(float));

    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, N * M * sizeof(float));
    cudaMalloc(&d_output, N * N * sizeof(float));

    srand(12345);
    for (int i = 0; i < N * M; i++) h_input[i] = (float)rand() / RAND_MAX;

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, N * M * sizeof(float), cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Use CUDA events for accurate timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Launch the kernel
    correlation_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N, M);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back from device to host
    cudaMemcpy(h_output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // The kernel only computes the upper triangle, so we fill the lower part on the CPU
    fill_symmetric(h_output, N);

    printf("Execution Time: %f seconds\n\n", milliseconds / 1000.0f);

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}