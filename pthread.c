#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Estructura para los datos de cada hilo
typedef struct {
    int thread_id;
    int row, col;       // Posición en la rejilla (i, j)
    int N;              // Tamaño de la matriz (N x N)
    int q;              // sqrt(num_hilos)
    int blockSize;      // N / q
    float **A, **B, **C;
} FoxData;

pthread_barrier_t barrier; // Barrera de sincronización

// Función para multiplicar submatrices (bloques)
void multiply_block(int size, int row_offset, int col_offset, int k_offset, 
                    float **A, float **B, float **C) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                C[row_offset + i][col_offset + j] += 
                    A[row_offset + i][k_offset + k] * B[k_offset + k][col_offset + j];
            }
        }
    }
}

// Función que ejecuta cada hilo
void* fox_algorithm(void* arg) {
    FoxData* data = (FoxData*)arg;
    int step, k_bar;
    
    for (step = 0; step < data->q; step++) {
        // 1. Determinar qué bloque de A se "difunde" en la fila
        k_bar = (data->row + step) % data->q;
        
        // 2. Multiplicar el bloque de A(row, k_bar) por B(k_bar, col)
        multiply_block(data->blockSize, 
                       data->row * data->blockSize, 
                       data->col * data->blockSize, 
                       k_bar * data->blockSize, 
                       data->A, data->B, data->C);

        // 3. Sincronización: Esperar a que todos terminen la etapa antes de seguir
        pthread_barrier_wait(&barrier);
    }
    
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    // Inicializar la semilla para los números aleatorios
    srand((unsigned int)time(NULL));
    
    // 1. Leer N desde la terminal (argv[1]). Si no se pone, por defecto es 1024.
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    
    // 2. Leer num_threads desde la terminal (argv[2]). Si no se pone, por defecto es 4.
    int num_threads = (argc > 2) ? atoi(argv[2]) : 4; 

    // Verificación de seguridad
    int q = (int)sqrt(num_threads);
    if (q * q != num_threads) {
        printf("Error: El número de hilos (%d) DEBE ser un cuadrado perfecto (4, 9, 16...)\n", num_threads);
        return -1;
    }
    
    int blockSize = N / q;

    // Reserva de memoria para matrices A, B y C
    float **A = (float**)malloc(N * sizeof(float*));
    float **B = (float**)malloc(N * sizeof(float*));
    float **C = (float**)malloc(N * sizeof(float*));
    
    for (int i = 0; i < N; i++) {
        A[i] = (float*)malloc(N * sizeof(float));
        B[i] = (float*)malloc(N * sizeof(float));
        C[i] = (float*)calloc(N, sizeof(float)); // Inicializar en 0
        
        for (int j = 0; j < N; j++) {
            // Generar números float aleatorios entre 1000.0 y 2000.0
            A[i][j] = 1000.0f + ((float)rand() / (float)RAND_MAX) * 1000.0f;
            B[i][j] = 1000.0f + ((float)rand() / (float)RAND_MAX) * 1000.0f;
        }
    }

    pthread_t threads[num_threads];
    FoxData thread_args[num_threads];
    pthread_barrier_init(&barrier, NULL, num_threads);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Crear hilos y asignarles su posición en la rejilla
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < q; j++) {
            int id = i * q + j;
            thread_args[id].thread_id = id;
            thread_args[id].row = i;
            thread_args[id].col = j;
            thread_args[id].N = N;
            thread_args[id].q = q;
            thread_args[id].blockSize = blockSize;
            thread_args[id].A = A;
            thread_args[id].B = B;
            thread_args[id].C = C;
            pthread_create(&threads[id], NULL, fox_algorithm, &thread_args[id]);
        }
    }

    // Esperar a los hilos
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Tipo de dato usado: float\n");
    printf("N=%d, Hilos=%d, Tiempo: %f segundos\n", N, num_threads, time_taken);

    // Limpieza
    pthread_barrier_destroy(&barrier);
    
    // Liberación de memoria de las matrices
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);

    return 0;
}