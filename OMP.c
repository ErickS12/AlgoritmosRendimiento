#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // Añadido para poder usar time(NULL)

// Función clásica para multiplicar los bloques (submatrices)
void multiply_block(int size, int row_offset, int col_offset, int k_offset, 
                    double **A, double **B, double **C) {
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
    // Inicializar la semilla para los números aleatorios
    srand((unsigned int)time(NULL));

    // 1. Leer N desde la terminal (argv[1]). Si no se pone, por defecto es 1024.
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    
    // 2. Leer num_threads desde la terminal (argv[2]). Si no se pone, por defecto es 4.
    int num_threads = (argc > 2) ? atoi(argv[2]) : 9; 

    // Verificación de seguridad
    int q = (int)sqrt(num_threads);
    if (q * q != num_threads) {
        printf("Error: El número de hilos (%d) DEBE ser un cuadrado perfecto (4, 9, 16...)\n", num_threads);
        return -1;
    }
    
    // Forzamos el número de hilos en OpenMP con la variable que leímos
    omp_set_num_threads(num_threads);
    
    int blockSize = N / q;

    // 1. Reserva de memoria dinámica para matrices A, B y C
    double **A = (double**)malloc(N * sizeof(double*));
    double **B = (double**)malloc(N * sizeof(double*));
    double **C = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        A[i] = (double*)malloc(N * sizeof(double));
        B[i] = (double*)malloc(N * sizeof(double));
        C[i] = (double*)calloc(N, sizeof(double)); // calloc inicializa en 0
        for (int j = 0; j < N; j++) {
            // Generar números double aleatorios entre 1000.0 y 2000.0
            A[i][j] = 1000.0 + ((double)rand() / RAND_MAX) * 1000.0;
            B[i][j] = 1000.0 + ((double)rand() / RAND_MAX) * 1000.0;
        }
    }

    // OpenMP tiene su propia función de reloj de alta precisión
    double start_time = omp_get_wtime();

    // 2. Inicio de la región paralela
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        // Calcular la posición (row, col) del hilo en la rejilla de Fox
        int row = tid / q;
        int col = tid % q;
        int k_bar;

        // 3. Ejecutar los pasos del algoritmo de Fox
        for (int step = 0; step < q; step++) {
            // Determinar qué bloque de A se "difunde" en esta etapa
            k_bar = (row + step) % q;
            
            // Multiplicar el bloque A(row, k_bar) por B(k_bar, col)
            multiply_block(blockSize, 
                           row * blockSize, 
                           col * blockSize, 
                           k_bar * blockSize, 
                           A, B, C);

            // Sincronización: Ningún hilo avanza a la siguiente etapa 
            // hasta que todos hayan terminado la multiplicación actual
            #pragma omp barrier
        }
    }

    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;

    printf("N=%d, Hilos=%d, Tiempo OpenMP: %f segundos\n", N, num_threads, time_taken);

    // Liberar memoria (importante para tamaños grandes como N=8192)
    for (int i = 0; i < N; i++) {
        free(A[i]); free(B[i]); free(C[i]);
    }
    free(A); free(B); free(C);

    return 0;
}