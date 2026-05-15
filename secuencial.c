#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    // Inicializar la semilla para los números aleatorios
    srand((unsigned int)time(NULL));

    // Permite recibir el tamaño N desde la terminal, por defecto usa 1024
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    
    // Reserva de memoria dinámica (arreglo 1D contiguo para mejorar la memoria caché)
    int *A = (int *)malloc(N * N * sizeof(int));
    int *B = (int *)malloc(N * N * sizeof(int));
    int *C = (int *)calloc(N * N, sizeof(int)); // calloc inicializa en 0
    
    // Llenar matrices con números aleatorios entre 1000 y 2000
    for (int i = 0; i < N * N; i++) {
        A[i] = 1000 + rand() % 1001;
        B[i] = 1000 + rand() % 1001;
    }
    
    // Configurar e iniciar el reloj
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // --- ALGORITMO SECUENCIAL CLÁSICO (1 Core, 1 Hilo) ---
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0; // Variable temporal para acelerar el cálculo
            for (int k = 0; k < N; k++) {
                // Notación 1D: Fila i * Ancho N + Columna actual
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    // ----------------------------------------------------
    
    // Detener el reloj
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("Tipo de dato usado: int\n");
    printf("N=%d, Tiempo Secuencial (1 Core): %f segundos\n", N, time_taken);
    
    // Limpiar la memoria RAM
    free(A);
    free(B);
    free(C);
    
    return 0;
}