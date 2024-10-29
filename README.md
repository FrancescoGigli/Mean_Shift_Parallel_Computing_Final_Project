
# Mean Shift Clustering Project

This project implements the Mean Shift clustering algorithm using three approaches:
1. **Sequential (Single-threaded)**
2. **Parallel (Multi-threaded with OpenMP)**
3. **CUDA (GPU-based)**

## Project Structure

### Source Files
- **main_sequential.cpp** - Runs the sequential version of Mean Shift.
- **main_parallel_openMP.cpp** - Executes the parallel version using OpenMP, allowing for multi-threaded execution.
- **main_parallel_cuda.cu** - Implements the CUDA version of Mean Shift for GPU acceleration.
- **MeanShift.cpp / MeanShift.h** - Core Mean Shift algorithm implementation. Contains functions for sequential, parallel (OpenMP), and, if defined, CUDA executions.
- **Point.cpp / Point.h** - Defines the 2D point structure and Euclidean distance calculation.
- **Utils.cpp / Utils.h** - Contains utility functions for reading CSV files, writing output, and managing file streams.

### Input and Output
- **Input:** CSV files containing points, located in the `generated_points` directory. Example file format:
  ```
  x,y
  1.5,2.3
  -0.5,4.2
  ...
  ```
- **Output:** Text files in the format `[configuration]_[kernel]_[sync type]_[size]_result.txt` (e.g., `parallel_flat_atomic_10000_result.txt`). These files contain details like execution time per iteration, total time, and cluster count.

## Compilation Instructions

### Prerequisites
- A **C++ compiler** with OpenMP support (e.g., g++).
- **CUDA toolkit** for CUDA compilation.

### Build Commands
1. **Sequential version:**
   ```bash
   g++ main_sequential.cpp Point.cpp MeanShift.cpp Utils.cpp -o main_sequential -std=c++11
   ```
2. **Parallel version (OpenMP):**
   ```bash
   g++ main_parallel_openMP.cpp Point.cpp MeanShift.cpp Utils.cpp -o main_parallel_openMP -std=c++11 -fopenmp
   ```
3. **CUDA version:**
   ```bash
   nvcc main_parallel_cuda.cu Point.cpp MeanShift.cpp Utils.cpp -o main_parallel_cuda -std=c++11
   ```

## Execution

Run each version with the following command format:

```bash
./[executable_name]
```

Example:
```bash
./main_sequential
./main_parallel_openMP
./main_parallel_cuda
```

## Program Configuration

### Parameters
The project includes configurations such as:
- Dataset sizes (`sizes` vector).
- Kernel types: `FLAT` or `GAUSSIAN`.
- Bandwidth and epsilon values, specified per kernel.
- Synchronization types (`sync_types`): `critical` and `atomic` (for parallel versions).

### Parallelization Details
- The parallel OpenMP version uses multiple threads, set with `omp_set_num_threads(num_threads)`.
- CUDA version uses GPU threads, with adjustable `threads_per_block` values for optimal performance.

## Output Description
Each run outputs a summary including:
- **Execution time** per iteration.
- **Total execution time.**
- **Number of clusters** identified.

Example:
```
Configuration: Parallel, Number of points: 50000, Kernel: flat, Sync: atomic
Number of threads used: 4
Iteration 1: 8973.594400 ms
...
Total time: 49214.144000 ms
Number of clusters: 2259
```

## License
This project is licensed under the MIT License.
