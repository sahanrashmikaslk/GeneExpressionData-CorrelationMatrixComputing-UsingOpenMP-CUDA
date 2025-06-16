#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

void print_matrix(const float* matrix, int total_rows, int total_cols, int rows_to_print, int cols_to_print, const char* title) {
    printf("--- %s ---\n", title);
    if (rows_to_print > total_rows) rows_to_print = total_rows;
    if (cols_to_print > total_cols) cols_to_print = total_cols;
    for (int i = 0; i < rows_to_print; i++) {
        for (int j = 0; j < cols_to_print; j++) {
            printf("%8.4f ", matrix[i * total_cols + j]);
        }
        printf("\n");
    }
    printf("---------------------------------------\n\n");
}

__global__ void correlation_kernel(float* input_matrix, float* output_matrix, int N, int M) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N && j >= i) {
        // USE DOUBLE PRECISION FOR ACCUMULATORS FOR NUMERICAL STABILITY
        double sum_X = 0.0, sum_Y = 0.0, sum_XY = 0.0;
        double sum_X2 = 0.0, sum_Y2 = 0.0;

        float* row_i = input_matrix + i * M;
        float* row_j = input_matrix + j * M;

        for (int k = 0; k < M; k++) {
            double val_i = (double)row_i[k];
            double val_j = (double)row_j[k];
            sum_X += val_i; sum_Y += val_j; sum_XY += val_i * val_j;
            sum_X2 += val_i * val_i; sum_Y2 += val_j * val_j;
        }

        double numerator = (double)M * sum_XY - sum_X * sum_Y;
        double denominator = sqrt(((double)M * sum_X2 - sum_X * sum_X) * ((double)M * sum_Y2 - sum_Y * sum_Y));
        
        output_matrix[i * N + j] = (denominator == 0.0) ? 1.0f : (float)(numerator / denominator);
    }
}

void fill_symmetric(float* matrix, int N) {
    for (int i = 0; i < N; i++) { for (int j = 0; j < i; j++) { matrix[j * N + i] = matrix[i * N + j]; } }
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "Usage: %s <N_variables> <M_samples>\n", argv[0]); return 1; }
    int N = atoi(argv[1]); int M = atoi(argv[2]);

    printf("Executing CUDA Version\n");
    printf("Matrix Size: N=%d, M=%d\n", N, M);

    float* h_input = (float*)malloc(N * M * sizeof(float));
    float* h_output = (float*)malloc(N * N * sizeof(float));
    if (!h_input || !h_output) { fprintf(stderr, "Host memory allocation failed!\n"); return 1; }

    srand(12345);
    for (int i = 0; i < N * M; i++) h_input[i] = (float)rand() / RAND_MAX;
    print_matrix(h_input, N, M, 8, 8, "Input Matrix (Snippet)");

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * M * sizeof(float));
    cudaMalloc(&d_output, N * N * sizeof(float));
    cudaMemcpy(d_input, h_input, N * M * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    correlation_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, N, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    fill_symmetric(h_output, N);
    
    printf("Execution Time: %f seconds\n", milliseconds / 1000.0f);
    print_matrix(h_output, N, N, 8, 8, "Output Correlation Matrix (Snippet)");

    free(h_input); free(h_output);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}