# Benchmark de Multiplicación de Matrices

Implementación del algoritmo de Fox para multiplicación de matrices cuadradas (N×N) en diferentes paradigmas de paralelismo, con medición comparativa de rendimiento.

## Descripción

Este proyecto implementa cinco variantes del algoritmo de Fox:

| Implementación | Modelo              | Características          |
|----------------|---------------------|--------------------------|
| **Secuencial** | Línea base          | Un único thread          |
| **OpenMP**     | Memoria compartida  | Directivas de compilador |
| **Pthreads**   | Memoria compartida  | Threads POSIX            |
| **MPI**        | Memoria distribuida | Paso de mensajes, topología 2D|
| **OpenCL**     | GPU                 | Kernels optimizados con tiling 16×16|

Los datos se representan como `int` por defecto, con opciones de compilación para `float` y `double`.

## Estructura del Proyecto

```
.
├── secuencial.c              # Versión secuencial
├── OMP.c                     # Versión OpenMP
├── pthread.c                 # Versión Pthreads
├── MPI.c                     # Versión MPI
├── matriz_opencl.c           # Versión OpenCL
└── README.md                 # Este archivo
```

## Compilación

### Secuencial
```bash
gcc secuencial.c -o secuencial_run
gcc secuencial.c -o secuencial_run -DUSE_FLOAT
gcc secuencial.c -o secuencial_run -DUSE_DOUBLE
```

### OpenMP
```bash
gcc OMP.c -o omp_run -fopenmp -lm
gcc OMP.c -o omp_run -fopenmp -DUSE_FLOAT -lm
gcc OMP.c -o omp_run -fopenmp -DUSE_DOUBLE -lm
```

### Pthreads
```bash
gcc pthread.c -o pthread_run -lpthread -lm
gcc pthread.c -o pthread_run -lpthread -DUSE_FLOAT -lm
gcc pthread.c -o pthread_run -lpthread -DUSE_DOUBLE -lm
```

### MPI
```bash
mpicc MPI.c -o mpi_run -lm
mpicc MPI.c -o mpi_run -DUSE_FLOAT -lm
mpicc MPI.c -o mpi_run -DUSE_DOUBLE -lm
```

### OpenCL
```bash
gcc matriz_opencl.c -o opencl_run -lOpenCL
gcc matriz_opencl.c -o opencl_run -lOpenCL -DUSE_FLOAT
gcc matriz_opencl.c -o opencl_run -lOpenCL -DUSE_DOUBLE
```

## Ejecución

### Secuencial
```bash
./secuencial_run [N]
./secuencial_run 1024
```

### OpenMP
```bash
./omp_run [N] [threads]
./omp_run 1024 9
```
El número de threads debe ser un cuadrado perfecto (4, 9, 16, 25, ...).

### Pthreads
```bash
./pthread_run [N] [threads]
./pthread_run 1024 4
```
El número de threads debe ser un cuadrado perfecto.

### MPI
```bash
mpirun -np [procesos] ./mpi_run [N]
mpirun -np 4 ./mpi_run 1024
mpirun --oversubscribe -np 9 ./mpi_run 1024
```
El número de procesos debe ser un cuadrado perfecto. Usar `--oversubscribe` si el número de procesos excede los núcleos disponibles.

### OpenCL
```bash
./opencl_run [N]
./opencl_run 1024
```
N debe ser múltiplo de 16 (tamaño del tile).

## Parámetros por Defecto

| Programa | N | Threads |
|----------|---|---------|
| secuencial | 1024 | - |
| OMP | 1024 | 9 |
| pthread | 1024 | 4 |
| MPI | 1024 | Configurable vía mpirun |
| opencl | 1024 | - |

## Dependencias

- GCC con soporte C99
- OpenMP (libgomp)
- libpthread
- OpenMPI o MPICH
- OpenCL (SDK según GPU: NVIDIA, AMD, Intel)

## Características Técnicas

- Tipo de dato: `int` (default), `float`, `double`
- Matrices de entrada: valores aleatorios en rango [1000, 2000]
- Medición de tiempo: `clock_gettime(CLOCK_MONOTONIC)`
- Almacenamiento en memoria: arreglos 1D para optimización de cache
- Topología MPI/paralela: rejilla 2D (q×q)
- **OpenCL:** El kernel GPU está incrustado dinámicamente en `matriz_opencl.c`. Implementa tiling 16×16 para optimización de memoria local.

## Referencias

Implementación del algoritmo de Fox para multiplicación de matrices distribuida.

