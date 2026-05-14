// kernel.cl
__kernel void matrix_mul(__global const float* A,
                         __global const float* B,
                         __global float* C,
                         const int N) {
    
    // Obtener la fila y columna que le toca a este "hilo" (Work-item) de la GPU
    int col = get_global_id(0);
    int row = get_global_id(1);

    // Asegurarse de no salir de los límites de la matriz
    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}