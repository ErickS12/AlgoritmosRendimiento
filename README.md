# Benchmark de Multiplicación de Matrices

Proyecto que compara el rendimiento de diferentes paradigmas de paralelismo en la multiplicación de matrices.

## 📋 Descripción

Este proyecto implementa la multiplicación de dos matrices cuadradas (N×N) usando distintas estrategias de paralelismo:

- **Secuencial**: Versión de un único hilo (baseline)
- **OpenMP (OMP)**: Paralelismo con directivas OpenMP
- **Pthreads**: Paralelismo con threads POSIX
- **MPI**: Paralelismo distribuido con Message Passing Interface
- **OpenCL**: Aceleración GPU

Cada implementación mide el tiempo de ejecución para comparar eficiencia y escalabilidad. En esta versión, las matrices y acumuladores de MPI, OpenMP, Pthreads y la versión secuencial se modificaron de double a int.

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

### Versión Secuencial (secuencial.c)
No requiere librerías especiales de paralelismo, solo la librería estándar.

**Con `int` (por defecto):**
```bash
gcc secuencial.c -o secuencial_run
```

**Con `float`:**
```bash
gcc secuencial.c -o secuencial_run -DUSE_FLOAT
```

**Con `double`:**
```bash
gcc secuencial.c -o secuencial_run -DUSE_DOUBLE
```

### OpenMP (OMP.c)
Requiere la bandera `-fopenmp` para habilitar las directivas de hilos del compilador GCC y `-lm` para las funciones matemáticas.

**Con `int` (por defecto):**
```bash
gcc OMP.c -o omp_run -fopenmp -lm
```

**Con `float`:**
```bash
gcc OMP.c -o omp_run -fopenmp -DUSE_FLOAT -lm
```

**Con `double`:**
```bash
gcc OMP.c -o omp_run -fopenmp -DUSE_DOUBLE -lm
```

### Pthreads (pthread.c)
Requiere enlazar explícitamente la librería de hilos nativa de Linux con `-lpthread` y `-lm`.

**Con `int` (por defecto):**
```bash
gcc pthread.c -o pthread_run -lpthread -lm
```

**Con `float`:**
```bash
gcc pthread.c -o pthread_run -lpthread -DUSE_FLOAT -lm
```

**Con `double`:**
```bash
gcc pthread.c -o pthread_run -lpthread -DUSE_DOUBLE -lm
```

### MPI (MPI.c)
Utiliza el compilador y el entorno de ejecución de tu distribución de MPI (como OpenMPI o MPICH). Requiere enlazar la librería matemática con `-lm`.

**Con `int` (por defecto):**
```bash
mpicc MPI.c -o mpi_run -lm
```

**Con `float`:**
```bash
mpicc MPI.c -o mpi_run -DUSE_FLOAT -lm
```

**Con `double`:**
```bash
mpicc MPI.c -o mpi_run -DUSE_DOUBLE -lm
```

### OpenCL (matriz_opencl.c)
Requiere tener instalados los SDKs/Drivers de OpenCL de tu hardware (Intel, AMD o NVIDIA) y enlazar la librería con `-lOpenCL`.

**Con `int` (por defecto):**
```bash
gcc matriz_opencl.c -o opencl_run -lOpenCL
```

**Con `float`:**
```bash
gcc matriz_opencl.c -o opencl_run -lOpenCL -DUSE_FLOAT
```

**Con `double`:**
```bash
gcc matriz_opencl.c -o opencl_run -lOpenCL -DUSE_DOUBLE
```

## 🚀 Ejecución

### Secuencial
**Sintaxis:** `./secuencial_run [N]`
```bash
./secuencial_run 1024
```

### OpenMP
**Sintaxis:** `./omp_run [N] [hilos]`

⚠️ **Nota:** El número de hilos debe ser un cuadrado perfecto (4, 9, 16, 25...) debido a la rejilla bidimensional del Algoritmo de Fox.
```bash
./omp_run 1024 9
```

### Pthreads
**Sintaxis:** `./pthread_run [N] [hilos]`

⚠️ **Nota:** El número de hilos debe ser un cuadrado perfecto (4, 9, 16, 25...).
```bash
./pthread_run 1024 4
```

### MPI
**Sintaxis:** `mpirun -np [procesos] ./mpi_run [N]`

⚠️ **Nota:** El número de procesos debe ser un cuadrado perfecto (4, 9, 16, 25...).

Si el número de procesos excede los núcleos disponibles en tu sistema, usa la flag `--oversubscribe`:
```bash
mpirun -np 4 ./mpi_run 1024
```

**Con oversubscribe (para sistemas con menos núcleos):**
```bash
mpirun --oversubscribe -np 9 ./mpi_run 1024
```

### OpenCL
**Sintaxis:** `./opencl_run [N]`

⚠️ **Nota:** El tamaño de la matriz N debe ser un múltiplo estricto de 16 debido al tamaño del bloque local/Tile.
```bash
./opencl_run 1024
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
- Las matrices se generan con valores aleatorios enteros entre 1000 y 2000
- El tiempo se mide usando `clock_gettime(CLOCK_MONOTONIC)` para máxima precisión
- Para obtener resultados confiables, ejecutar múltiples veces y promediar

## 📝 Autor

Códigos de algoritmos y análisis de rendimiento

## 📄 Licencia

Libre para uso educativo y de investigación
