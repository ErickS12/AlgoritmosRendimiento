#include <pthread.h>
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

typedef struct {
    int thread_id;
    int row, col, N, q, blockSize;
    DTYPE **A, **B, **C;
} FoxData;

pthread_barrier_t barrier;

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

void* fox_algorithm(void* arg) {
    FoxData* data = (FoxData*)arg;
    int k_bar;
    
    for (int step = 0; step < data->q; step++) {
        k_bar = (data->row + step) % data->q;
        multiply_block(data->blockSize, data->row * data->blockSize, 
                       data->col * data->blockSize, k_bar * data->blockSize, 
                       data->A, data->B, data->C);
        pthread_barrier_wait(&barrier);
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    srand((unsigned int)time(NULL));
    
    // N e hilos desde terminal
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int num_threads = (argc > 2) ? atoi(argv[2]) : 4; 

    int q = (int)sqrt(num_threads);
    if (q * q != num_threads) {
        printf("Error: El número de hilos debe ser un cuadrado perfecto.\n");
        return -1;
    }
    
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

    pthread_t threads[num_threads];
    FoxData thread_args[num_threads];
    pthread_barrier_init(&barrier, NULL, num_threads);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            int id = i * q + j;
            thread_args[id] = (FoxData){id, i, j, N, q, blockSize, A, B, C};
            pthread_create(&threads[id], NULL, fox_algorithm, &thread_args[id]);
        }
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    
    printf("Tipo de dato usado: %s\n", DTYPE_NAME);
    printf("N=%d, Hilos=%d, Tiempo Pthreads: %f segundos\n", N, num_threads, 
          (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9);

    pthread_barrier_destroy(&barrier);
    for (int i = 0; i < N; i++) { free(A[i]); free(B[i]); free(C[i]); }
    free(A); free(B); free(C);

    return 0;
}