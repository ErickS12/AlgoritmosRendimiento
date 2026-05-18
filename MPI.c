#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- SELECCIÓN DINÁMICA DEL TIPO DE DATO ---
#if defined(USE_FLOAT)
    typedef float DTYPE;
    #define MPI_DTYPE MPI_FLOAT
    #define DTYPE_NAME "float"
#elif defined(USE_DOUBLE)
    typedef double DTYPE;
    #define MPI_DTYPE MPI_DOUBLE
    #define DTYPE_NAME "double"
#else
    typedef int DTYPE;
    #define MPI_DTYPE MPI_INT
    #define DTYPE_NAME "int"
#endif

void multiply_block(int size, DTYPE *A, DTYPE *B, DTYPE *C) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    srand((unsigned int)time(NULL) + rank);

    // N desde terminal
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int q = (int)sqrt(num_procs);

    if (q * q != num_procs) {
        if (rank == 0) printf("Error: El número de procesos debe ser un cuadrado perfecto.\n");
        MPI_Finalize();
        return -1;
    }

    int blockSize = N / q;
    MPI_Comm grid_comm;
    int dims[2] = {q, q};
    int periods[2] = {1, 1};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    int coords[2];
    MPI_Cart_coords(grid_comm, rank, 2, coords);
    int my_row = coords[0];
    int my_col = coords[1];

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(grid_comm, my_row, my_col, &row_comm);
    MPI_Comm_split(grid_comm, my_col, my_row, &col_comm);

    int block_elements = blockSize * blockSize;
    DTYPE *local_A = (DTYPE*)malloc(block_elements * sizeof(DTYPE));
    DTYPE *local_B = (DTYPE*)malloc(block_elements * sizeof(DTYPE));
    DTYPE *local_C = (DTYPE*)calloc(block_elements, sizeof(DTYPE));
    DTYPE *temp_A  = (DTYPE*)malloc(block_elements * sizeof(DTYPE));

    for (int i = 0; i < block_elements; i++) {
        #if defined(USE_FLOAT) || defined(USE_DOUBLE)
            local_A[i] = (DTYPE)1000.0 + ((DTYPE)rand() / RAND_MAX) * 1000.0;
            local_B[i] = (DTYPE)1000.0 + ((DTYPE)rand() / RAND_MAX) * 1000.0;
        #else
            local_A[i] = (DTYPE)(1000 + rand() % 1001);
            local_B[i] = (DTYPE)(1000 + rand() % 1001);
        #endif
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    int source, dest;
    MPI_Cart_shift(grid_comm, 0, -1, &source, &dest); 

    for (int step = 0; step < q; step++) {
        int bcast_root = (my_row + step) % q;
        if (my_col == bcast_root) {
            for(int i=0; i<block_elements; i++) temp_A[i] = local_A[i];
        }

        // Se usa la macro MPI_DTYPE
        MPI_Bcast(temp_A, block_elements, MPI_DTYPE, bcast_root, row_comm);
        multiply_block(blockSize, temp_A, local_B, local_C);
        MPI_Sendrecv_replace(local_B, block_elements, MPI_DTYPE, 
                             dest, 0, source, 0, grid_comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Tipo de dato usado: %s\n", DTYPE_NAME);
        printf("N=%d, Procesos=%d, Tiempo MPI: %f segundos\n", N, num_procs, end_time - start_time);
    }

    free(local_A); free(local_B); free(local_C); free(temp_A);
    MPI_Comm_free(&row_comm); MPI_Comm_free(&col_comm); MPI_Comm_free(&grid_comm);
    MPI_Finalize();
    return 0;
}