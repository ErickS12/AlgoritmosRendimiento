#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#if defined(USE_FLOAT)
    typedef float DTYPE;
    #define DTYPE_NAME "float"
#elif defined(USE_DOUBLE)
    typedef double DTYPE;
    #define DTYPE_NAME "double"
#else
    typedef int DTYPE;
    #define DTYPE_NAME "int"
#endif

void multiply_block(int size, int row_offset, int col_offset, int k_offset, 
                    DTYPE **A, DTYPE **B, DTYPE **C) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                C[row_offset + i][col_offset + j] += 
                    A[row_offset + i][k_offset + k] * B[k_offset + k][col_offset + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    srand((unsigned int)time(NULL));

    // N e hilos desde terminal
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int num_threads = (argc > 2) ? atoi(argv[2]) : 9; 

    int q = (int)sqrt(num_threads);
    if (q * q != num_threads) {
        printf("Error: El número de hilos debe ser un cuadrado perfecto.\n");
        return -1;
    }
    
    omp_set_num_threads(num_threads);
    int blockSize = N / q;

    DTYPE **A = (DTYPE**)malloc(N * sizeof(DTYPE*));
    DTYPE **B = (DTYPE**)malloc(N * sizeof(DTYPE*));
    DTYPE **C = (DTYPE**)malloc(N * sizeof(DTYPE*));
    
    for (int i = 0; i < N; i++) {
        A[i] = (DTYPE*)malloc(N * sizeof(DTYPE));
        B[i] = (DTYPE*)malloc(N * sizeof(DTYPE));
        C[i] = (DTYPE*)calloc(N, sizeof(DTYPE));
        for (int j = 0; j < N; j++) {
            #if defined(USE_FLOAT) || defined(USE_DOUBLE)
                A[i][j] = (DTYPE)1000.0 + ((DTYPE)rand() / RAND_MAX) * 1000.0;
                B[i][j] = (DTYPE)1000.0 + ((DTYPE)rand() / RAND_MAX) * 1000.0;
            #else
                A[i][j] = (DTYPE)(1000 + rand() % 1001);
                B[i][j] = (DTYPE)(1000 + rand() % 1001);
            #endif
        }
    }

    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int row = tid / q;
        int col = tid % q;
        int k_bar;

        for (int step = 0; step < q; step++) {
            k_bar = (row + step) % q;
            multiply_block(blockSize, row * blockSize, col * blockSize, k_bar * blockSize, A, B, C);
            #pragma omp barrier
        }
    }

    double end_time = omp_get_wtime();
    
    printf("Tipo de dato usado: %s\n", DTYPE_NAME);
    printf("N=%d, Hilos=%d, Tiempo OpenMP: %f segundos\n", N, num_threads, end_time - start_time);

    for (int i = 0; i < N; i++) {
        free(A[i]); free(B[i]); free(C[i]);
    }
    free(A); free(B); free(C);

    return 0;
}