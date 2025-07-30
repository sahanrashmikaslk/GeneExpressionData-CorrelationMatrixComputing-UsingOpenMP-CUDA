#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Function to calculate RMSE between two correlation matrices
double calculate_rmse(float *matrix1, float *matrix2, int N)
{
    double sum_squared_diff = 0.0;
    int total_elements = N * N;

    for (int i = 0; i < total_elements; i++)
    {
        double diff = matrix1[i] - matrix2[i];
        sum_squared_diff += diff * diff;
    }

    return sqrt(sum_squared_diff / total_elements);
}

// Function to load matrix from binary file
int load_matrix_from_file(const char *filename, float **matrix, int N)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        return -1;
    }

    *matrix = (float *)malloc(N * N * sizeof(float));
    if (!*matrix)
    {
        fclose(file);
        return -1;
    }

    size_t read_count = fread(*matrix, sizeof(float), N * N, file);
    fclose(file);

    if (read_count != N * N)
    {
        free(*matrix);
        return -1;
    }

    return 0;
}

// Function to save matrix to binary file
int save_matrix_to_file(const char *filename, float *matrix, int N)
{
    FILE *file = fopen(filename, "wb");
    if (!file)
    {
        return -1;
    }

    size_t written = fwrite(matrix, sizeof(float), N * N, file);
    fclose(file);

    return (written == N * N) ? 0 : -1;
}

// Function to compare with reference matrix and calculate RMSE
void verify_accuracy(float *result_matrix, int N, const char *implementation_name)
{
    char reference_file[] = "serial_reference.bin";
    char result_file[256];
    snprintf(result_file, sizeof(result_file), "%s_result.bin", implementation_name);

    // Save current result
    save_matrix_to_file(result_file, result_matrix, N);

    // Try to load reference matrix
    float *reference_matrix = NULL;
    if (load_matrix_from_file(reference_file, &reference_matrix, N) == 0)
    {
        // Calculate RMSE
        double rmse = calculate_rmse(reference_matrix, result_matrix, N);

        printf("\n═══════════════════════════════════════════════════════════════\n");
        printf("                    ACCURACY VERIFICATION\n");
        printf("═══════════════════════════════════════════════════════════════\n");
        printf("Implementation: %s\n", implementation_name);
        printf("Matrix Size: %d × %d\n", N, N);
        printf("RMSE vs Serial: %.2e\n", rmse);

        if (rmse < 1e-12)
        {
            printf("Accuracy: EXCELLENT (RMSE < 1e-12)\n");
        }
        else if (rmse < 1e-6)
        {
            printf("Accuracy: VERY GOOD (RMSE < 1e-6)\n");
        }
        else if (rmse < 1e-3)
        {
            printf("Accuracy: ACCEPTABLE (RMSE < 1e-3)\n");
        }
        else
        {
            printf("Accuracy: POOR (RMSE ≥ 1e-3)\n");
        }
        printf("═══════════════════════════════════════════════════════════════\n");

        free(reference_matrix);
    }
    else
    {
        // If this is serial implementation, save as reference
        if (strcmp(implementation_name, "serial") == 0)
        {
            save_matrix_to_file(reference_file, result_matrix, N);
            printf("\n💾 Serial result saved as reference for accuracy verification\n");
        }
        else
        {
            printf("\n No reference matrix found for accuracy verification\n");
            printf("   Run serial implementation first to generate reference\n");
        }
    }
}
