# Benchmark de Multiplicación de Matrices

Proyecto que compara el rendimiento de diferentes paradigmas de paralelismo en la multiplicación de matrices.

## 📋 Descripción

Este proyecto implementa la multiplicación de dos matrices cuadradas (N×N) usando distintas estrategias de paralelismo:

- **Secuencial**: Versión de un único hilo (baseline)
- **OpenMP (OMP)**: Paralelismo con directivas OpenMP
- **Pthreads**: Paralelismo con threads POSIX
- **MPI**: Paralelismo distribuido con Message Passing Interface
- **OpenCL**: Aceleración GPU

Cada implementación mide el tiempo de ejecución para comparar eficiencia y escalabilidad. En esta versión, las matrices y acumuladores de MPI, OpenMP, Pthreads y la versión secuencial se modificaron de double a float.

## 🏗️ Estructura del Proyecto

```
.
├── secuencial.c              # Versión secuencial
├── OMP.c                     # Versión OpenMP
├── pthread.c                 # Versión Pthreads
├── MPI.c                     # Versión MPI
├── matriz_opencl.c           # Versión OpenCL
├── kernel.cl                 # Kernel GPU para OpenCL
├── benchmark_gpu             # Script/binary para benchmarks GPU
└── README.md                 # Este archivo
```

## 🔨 Compilación

### Versión Secuencial
```bash
gcc -O2 -o secuencial secuencial.c -lm
```

### OpenMP
```bash
gcc -O2 -fopenmp -o OMP OMP.c -lm
```

### Pthreads
```bash
gcc -O2 -o pthread pthread.c -lpthread -lm
```

### MPI
```bash
mpicc -O2 -o MPI MPI.c -lm
```

### OpenCL
```bash
gcc -O2 -o matriz_opencl matriz_opencl.c -lOpenCL -lm
```

## 🚀 Ejecución

Todos los programas aceptan el tamaño de matriz como argumento (por defecto 1024):

### Secuencial
```bash
./secuencial 1024
```

### OpenMP
```bash
./OMP 1024 9
# Parámetros: ./OMP [tamaño_matriz] [num_threads]
# num_threads DEBE ser un número cuadrado perfecto (4, 9, 16, 25...)
```

### Pthreads
```bash
./pthread 1024 9
# Parámetros: ./pthread [tamaño_matriz] [num_threads]
# num_threads DEBE ser un número cuadrado perfecto
```

### MPI
```bash
mpirun -np 9 ./MPI 1024
# Parámetros: -np [num_procesos] - DEBE ser un número cuadrado perfecto
```

### OpenCL
```bash
./matriz_opencl 1024
```

## 📊 Parámetros

| Programa | Parámetro 1 | Parámetro 2 |
|----------|------------|------------|
| secuencial | Tamaño N (default: 1024) | - |
| OMP | Tamaño N (default: 1024) | Threads (default: 9) |
| pthread | Tamaño N (default: 1024) | Threads (default: 4) |
| MPI | Tamaño N (default: 1024) | Procesos (via mpirun) |
| matriz_opencl | Tamaño N (default: 1024) | - |

## ⚙️ Requisitos

- **GCC** con soporte C99
- **OpenMP** (libgomp)
- **libpthread** (POSIX threads)
- **OpenMPI** o MPICH (para MPI)
- **OpenCL** (headers y library para GPU)
- **GNU Make** (opcional, si hay Makefile)

## 📈 Notas sobre Rendimiento

- Los algoritmos están optimizados para **cache locality** usando arreglos 1D
- Las matrices se generan con valores aleatorios flotantes entre 1000.0 y 2000.0
- El tiempo se mide usando `clock_gettime(CLOCK_MONOTONIC)` para máxima precisión
- Para obtener resultados confiables, ejecutar múltiples veces y promediar

## 📝 Autor

Códigos de algoritmos y análisis de rendimiento

## 📄 Licencia

Libre para uso educativo y de investigación
