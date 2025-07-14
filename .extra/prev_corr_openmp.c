#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>



//to generate a matrix of random floats
void generate_input_matrix(float* matrix, int n, int m) {
    for (int i = 0; i < n * m; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Function to compute the Pearson correlation matrix with OpenMP

void openmp_correlation(float* input_matrix, float* output_matrix, int N, int M) {

    #pragma omp parallel for schedule(dynamic)
    
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            float sum_X = 0.0f, sum_Y = 0.0f, sum_XY = 0.0f;
            float sum_X2 = 0.0f, sum_Y2 = 0.0f;

            float* row_i = input_matrix + i * M;
            float* row_j = input_matrix + j * M;

            for (int k = 0; k < M; k++) {
                sum_X += row_i[k];
                sum_Y += row_j[k];
                sum_XY += row_i[k] * row_j[k];
                sum_X2 += row_i[k] * row_i[k];
                sum_Y2 += row_j[k] * row_j[k];
            }

            float numerator = (float)M * sum_XY - sum_X * sum_Y;
            float denominator = sqrtf(((float)M * sum_X2 - sum_X * sum_X) * ((float)M * sum_Y2 - sum_Y * sum_Y));
            
            output_matrix[i * N + j] = (denominator == 0.0f) ? 1.0f : numerator / denominator;
            output_matrix[j * N + i] = output_matrix[i * N + j];
        }
    }
}

// to print the input matrix
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



int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N_variables> <M_samples>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int M = atoi(argv[2]);

    printf("Executing OpenMP Version\n");
    printf("Matrix Size: N=%d, M=%d\n", N, M);
    printf("Num threads: %d\n", omp_get_max_threads());

    float* input_matrix = (float*)malloc(N * M * sizeof(float));
    float* output_matrix = (float*)malloc(N * N * sizeof(float));

    if (!input_matrix || !output_matrix) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }

    
    srand(12345);
    generate_input_matrix(input_matrix, N, M);
    
    print_matrix(input_matrix, N, M, 8, 8, "Input Matrix (Snippet)");

    double start_time = omp_get_wtime();
    openmp_correlation(input_matrix, output_matrix, N, M);
    double end_time = omp_get_wtime();

 
   
    
    print_matrix(output_matrix, N, N, 8, 8, "Output Correlation Matrix (Snippet)");

    printf("Execution Time: %f seconds\n", end_time - start_time);

    free(input_matrix);
    free(output_matrix);

    return 0;
}