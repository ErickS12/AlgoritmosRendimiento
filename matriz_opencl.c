#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Tamaño del bloque (Tile Size). 16x16 es el estándar para GPUs.
#define TS 16 

// 1. KERNEL OPTIMIZADO: Multiplicación por Bloques (Tiled Matrix Multiplication)
const char *kernelSource = 
"#define TS 16\n"
"__kernel void matrix_mul_tiled(__global const float* A,\n"
"                               __global const float* B,\n"
"                               __global float* C,\n"
"                               const int N) {\n"
"    // Identificadores globales\n"
"    int row = get_global_id(1);\n"
"    int col = get_global_id(0);\n"
"    \n"
"    // Identificadores locales\n"
"    int local_row = get_local_id(1);\n"
"    int local_col = get_local_id(0);\n"
"    \n"
"    // Memoria local (ultrarrápida) para los sub-bloques\n"
"    __local float Asub[TS][TS];\n"
"    __local float Bsub[TS][TS];\n"
"    \n"
"    float sum = 0.0f;\n"
"    \n"
"    // Recorrer los bloques necesarios\n"
"    int numTiles = N / TS;\n"
"    for (int t = 0; t < numTiles; t++) {\n"
"        // Cargar un elemento en memoria local\n"
"        Asub[local_row][local_col] = A[row * N + (t * TS + local_col)];\n"
"        Bsub[local_row][local_col] = B[(t * TS + local_row) * N + col];\n"
"        \n"
"        // Sincronizar\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"        \n"
"        // Multiplicar sub-bloques\n"
"        for (int k = 0; k < TS; k++) {\n"
"            sum += Asub[local_row][k] * Bsub[k][local_col];\n"
"        }\n"
"        \n"
"        // Sincronizar antes del siguiente bloque\n"
"        barrier(CLK_LOCAL_MEM_FENCE);\n"
"    }\n"
"    \n"
"    // Escribir el resultado final\n"
"    C[row * N + col] = sum;\n"
"}\n";

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    // N debe ser múltiplo de TS (16) para este kernel simplificado
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    if (N % TS != 0) {
        printf("Error: N debe ser múltiplo de %d\n", TS);
        return 1;
    }

    size_t size = N * N * sizeof(float);
    cl_int err;

    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C = (float*)malloc(size);

    for (int i = 0; i < N * N; i++) {
        A[i] = 1000.0f + ((float)rand() / RAND_MAX) * 1000.0f;
        B[i] = 1000.0f + ((float)rand() / RAND_MAX) * 1000.0f;
    }

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, A, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, B, &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &err);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    // Verificación de errores de compilación del kernel
    if (err != CL_SUCCESS) {
        char build_log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, NULL);
        printf("Error en clBuildProgram:\n%s\n", build_log);
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_mul_tiled", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    size_t global_work_size[2] = {N, N}; 
    size_t local_work_size[2] = {TS, TS}; 

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    clFinish(queue);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("N=%d, Tiempo OpenCL Tiled (Benchmark GPU Intel): %f segundos\n", N, time_taken);

    clReleaseMemObject(bufferA); clReleaseMemObject(bufferB); clReleaseMemObject(bufferC);
    clReleaseKernel(kernel); clReleaseProgram(program);
    clReleaseCommandQueue(queue); clReleaseContext(context);
    free(A); free(B); free(C);

    return 0;
}