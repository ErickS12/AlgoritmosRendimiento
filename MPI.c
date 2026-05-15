#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> // Necesario para time(NULL)

// Función estándar de multiplicación local
void multiply_block(int size, int *A, int *B, int *C) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                // Notación 1D para matrices contiguas en memoria
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, num_procs;
    
    // MPI ya usa argc y argv internamente en Init, por eso se los pasamos
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); // Aquí MPI descubre cuántos procesos lanzaste

    // Inicializar la semilla combinando el tiempo y el rank del proceso
    srand((unsigned int)time(NULL) + rank);

    // 1. Leer N desde la terminal (argv[1]). Si no se pone, por defecto es 1024.
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    
    int q = (int)sqrt(num_procs);

    // Verificación de seguridad
    if (q * q != num_procs) {
        if (rank == 0) printf("Error: El número de procesos (%d) debe ser un cuadrado perfecto (4, 9, 16...).\n", num_procs);
        MPI_Finalize();
        return -1;
    }

    int blockSize = N / q;

    // 1. Crear la topología Cartesiana 2D
    MPI_Comm grid_comm;
    int dims[2] = {q, q};
    int periods[2] = {1, 1}; // Habilitar rotación circular (toroide)
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    // Obtener coordenadas del proceso actual en la cuadrícula (row, col)
    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    // Crear comunicadores separados para filas y columnas
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(grid_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(grid_comm, my_col, my_row, &col_comm);

    // Asignar memoria para los bloques LOCALES (contiguos en memoria)
    int block_elements = blockSize * blockSize;
    int *local_A = (int*)malloc(block_elements * sizeof(int));
    int *local_B = (int*)malloc(block_elements * sizeof(int));
    int *local_C = (int*)calloc(block_elements, sizeof(int));
    int *temp_A  = (int*)malloc(block_elements * sizeof(int)); // Buffer para recibir A

    // Inicializar bloques con datos aleatorios entre 1000 y 2000
    for (int i = 0; i < block_elements; i++) {
        local_A[i] = 1000 + rand() % 1001;
        local_B[i] = 1000 + rand() % 1001;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // 2. Ejecutar el Algoritmo de Fox
    int source, dest;
    // Calcular vecinos para la rotación de la matriz B (hacia arriba)
    MPI_Cart_shift(grid_comm, 0, -1, &source, &dest); 

    for (int step = 0; step < q; step++) {
        // a) Determinar quién hace el Broadcast de A en esta fila
        int bcast_root = (my_row + step) % q;

        // Si soy el root de esta etapa, copio mi A al buffer temporal
        if (my_col == bcast_root) {
            for(int i=0; i<block_elements; i++) temp_A[i] = local_A[i];
        }

        // b) Difundir el bloque de A a través del comunicador de fila
        MPI_Bcast(temp_A, block_elements, MPI_INT, bcast_root, row_comm);

        // c) Multiplicar el bloque recibido de A por el bloque local de B
        multiply_block(blockSize, temp_A, local_B, local_C);

        // d) Rotar el bloque local de B hacia el proceso de arriba
        // MPI_Sendrecv_replace envía el buffer y lo sobrescribe con lo que recibe
        MPI_Sendrecv_replace(local_B, block_elements, MPI_INT, 
                             dest, 0, source, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Tipo de dato usado: int\n");
        printf("N=%d, Procesos=%d, Tiempo MPI: %f segundos\n", N, num_procs, end_time - start_time);
    }

    // Limpieza
    free(local_A); free(local_B); free(local_C); free(temp_A);
    MPI_Comm_free(&row_comm); MPI_Comm_free(&col_comm); MPI_Comm_free(&grid_comm);
    MPI_Finalize();
    return 0;
}