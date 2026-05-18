#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// --- SELECCIÓN DINÁMICA DEL TIPO DE DATO ---
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

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    // N desde terminal
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    
    DTYPE *A = (DTYPE *)malloc(N * N * sizeof(DTYPE));
    DTYPE *B = (DTYPE *)malloc(N * N * sizeof(DTYPE));
    DTYPE *C = (DTYPE *)calloc(N * N, sizeof(DTYPE));
    
    for (int i = 0; i < N * N; i++) {
        #if defined(USE_FLOAT) || defined(USE_DOUBLE)
            A[i] = (DTYPE)1000.0 + ((DTYPE)rand() / RAND_MAX) * 1000.0;
            B[i] = (DTYPE)1000.0 + ((DTYPE)rand() / RAND_MAX) * 1000.0;
        #else
            A[i] = (DTYPE)(1000 + rand() % 1001);
            B[i] = (DTYPE)(1000 + rand() % 1001);
        #endif
    }
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            DTYPE sum = 0; 
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Tipo de dato usado: %s\n", DTYPE_NAME);
    printf("N=%d, Tiempo Secuencial: %f segundos\n", N, time_taken);
    
    free(A); free(B); free(C);
    return 0;
}