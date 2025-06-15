#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Helper function to generate a matrix of random floats
void generate_input_matrix(float* matrix, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Function to compute the Pearson correlation matrix serially
void serial_correlation(float* input_matrix, float* output_matrix, int N, int M) {
    // Iterate over each pair of variables (rows)
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            float sum_X = 0.0, sum_Y = 0.0, sum_XY = 0.0;
            float sum_X2 = 0.0, sum_Y2 = 0.0;

            // Pointers to the start of rows i and j
            float* row_i = input_matrix + i * M;
            float* row_j = input_matrix + j * M;

            for (int k = 0; k < M; k++) {
                sum_X += row_i[k];
                sum_Y += row_j[k];
                sum_XY += row_i[k] * row_j[k];
                sum_X2 += row_i[k] * row_i[k];
                sum_Y2 += row_j[k] * row_j[k];
            }

            float numerator = M * sum_XY - sum_X * sum_Y;
            float denominator = sqrt((M * sum_X2 - sum_X * sum_X) * (M * sum_Y2 - sum_Y * sum_Y));
            
            // Avoid division by zero
            output_matrix[i * N + j] = (denominator == 0) ? 1.0 : numerator / denominator;
            output_matrix[j * N + i] = output_matrix[i * N + j]; // Symmetric matrix
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

    printf("Executing Serial Version\n");
    printf("Matrix Size: N=%d, M=%d\n", N, M);

    // Allocate memory
    float* input_matrix = (float*)malloc(N * M * sizeof(float));
    float* output_matrix = (float*)malloc(N * N * sizeof(float));

    if (!input_matrix || !output_matrix) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    // Generate data and run computation
    srand(time(NULL));
    generate_input_matrix(input_matrix, N, M);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    serial_correlation(input_matrix, output_matrix, N, M);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Execution Time: %f seconds\n\n", time_spent);

    // Free memory
    free(input_matrix);
    free(output_matrix);

    return 0;
}