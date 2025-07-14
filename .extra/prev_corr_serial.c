#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void generate_input_matrix(float* matrix, int n, int m) {
    srand(12345);
    for (int i = 0; i < n * m; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX;
    }
}

void print_matrix(const float* matrix, int rows, int cols, const char* title) {
    printf("--- %s (Snippet) ---\n", title);
    int print_rows = (rows < 8) ? rows : 8;
    int print_cols = (cols < 8) ? cols : 8;
    for (int i = 0; i < print_rows; i++) {
        for (int j = 0; j < print_cols; j++) {
            printf("%8.4f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("---------------------------------------\n\n");
}

void serial_correlation(float* input_matrix, float* output_matrix, int N, int M) {
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            if (i == j) {
                output_matrix[i * N + j] = 1.0f;
                continue;
            }
            // Use double for stable calculations
            double sum_X = 0.0, sum_Y = 0.0, sum_XY = 0.0;
            double sum_X2 = 0.0, sum_Y2 = 0.0;
            float* row_i = input_matrix + i * M;
            float* row_j = input_matrix + j * M;

            for (int k = 0; k < M; k++) {
                sum_X += row_i[k];
                sum_Y += row_j[k];
                sum_XY += row_i[k] * row_j[k];
                sum_X2 += row_i[k] * row_i[k];
                sum_Y2 += row_j[k] * row_j[k];
            }

            double numerator = (double)M * sum_XY - sum_X * sum_Y;
            double denominator = sqrt(((double)M * sum_X2 - sum_X * sum_X) * ((double)M * sum_Y2 - sum_Y * sum_Y));
            
            float result = (denominator == 0.0) ? 0.0f : (float)(numerator / denominator);
            output_matrix[i * N + j] = result;
            output_matrix[j * N + i] = result;
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "Usage: %s <N_variables> <M_samples>\n", argv[0]); return 1; }
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    printf("=== Serial Pearson Correlation ===\n");
    printf("Matrix Size: N=%d, M=%d\n\n", N, M);

    float* input_matrix = (float*)malloc(N * M * sizeof(float));
    float* output_matrix = (float*)malloc(N * N * sizeof(float));
    if (!input_matrix || !output_matrix) { fprintf(stderr, "Memory allocation failed!\n"); return 1; }

    generate_input_matrix(input_matrix, N, M);
    print_matrix(input_matrix, N, M, "Input Data");

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    serial_correlation(input_matrix, output_matrix, N, M);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Execution Time: %.4f seconds\n", time_spent);
    print_matrix(output_matrix, N, N, "Output Correlation Matrix");

    free(input_matrix);
    free(output_matrix);
    return 0;
}