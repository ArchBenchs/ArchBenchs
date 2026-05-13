ArchBenchs_LU_Decomposition
============

![C++17](https://img.shields.io/badge/C%2B%2B-17-red.svg)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20riscv-lightgrey.svg)
![CMake](https://img.shields.io/badge/CMake-3.8+-064F8C.svg)
![OpenMP](https://img.shields.io/badge/OpenMP-supported-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

The project ArchBenchs_LU_Decomposition is a part of ArchBenchs benchmarks and provides a high-performance implementation of LU decomposition for dense square matrices. It includes both a naive algorithm and a blocked (tiled) version optimized with OpenMP parallelism and cache-friendly memory access patterns. The project is designed for benchmarking, testing, and comparing LU decomposition performance across different architectures (x86, RISC-V) and reference libraries (Eigen, Intel MKL).

Getting Started
---------------
This repository doesn't use git submodules. To clone and use the repository:
```bash
git clone -b LU_Decomposition --single-branch https://github.com/ArchBenchs/ArchBenchs.git
```
The source files are located in the source/ directory. The project uses CMake for building and supports optional libraries used for reference comparisons: Eigen 5.0.0 is included into project in eigen-5.0.0/ directory and Intel MKL is supported (the library isn't included, it need to be installed in the system). The installation process is described below.

Installing
----------

### Dependencies
- `C++17` compatible compiler (GCC or Clang (for x86 or for RISC-V), Intel ICPX)
- [`CMake`](https://cmake.org/) 3.8 or higher
- `OpenMP` (for parallelization)
- (Optional) [`Intel MKL`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) for reference comparisons

### Build
Use CMake to build project. Available toolchain files, which are used in development:
```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/risk-V_gcc.cmake
```
```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/risk-V_clang.cmake
```
```bash
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchains/x86_icpx2023.cmake
```
Available CMake options:
- `-DBLOCK_SIZE=<N>` - set size of block used in block LU algorithm, 32 is recommended for RISC-V (default 64);
- `-DPRINT_BLOCK_TIMES=1`	- (parameter was used during optimization) enables measuring part's times of block LU algorithm;
- `-DREFERENCE_TEST=eigen` or `-DREFERENCE_TEST=mkl` - set library to compare results with. Works with exactly same matrices;
- `-DTYPE=<type>`	- set type of values in matrices (default double).

Running
----------
Basic usage (on Linux):
```bash
./LU_Benchmark --size 1000 --count 1
```
Command-line options:
- `--help` – show full help
- `--size <N>` – matrix size for time tests (default: 1000)
- `--count <N>` – number of test repetitions (default: 1)
- `--out <file>` – redirect output to a text file
- `--wt` / `--workability_tests` – run correctness tests on small matrices
- `--dac` / `--disable_accuracy_check` – skip result verification in time tests
- `--ri` / `--random_initialization` – use random matrices (range [1e-6, 1e6]) instead of diagonally dominant ones

Documentation
----------
### Algorithm Overview
The implementation provides two LU decomposition variants:

1. **Naive (`get_LU`)** – simple triple loop with OpenMP.
2. **Blocked (`block_get_LU`)** – tiled algorithm with block size `BLOCK_SIZE`.

The blocked version includes:

- L11 & U11 decomposition
- L21 computation  
- U12 computation
- Schur complement update: `A22 -= L21 × U12`

The blocked algorithm has improved cache locality and is well-suited for large matrices.

### Code Structure

- `square_matrix.h/cpp` – dense square matrix class with aligned memory allocation, arithmetic operators, norms, and I/O.
- `decomposer_lu.h/cpp` – LU decomposition implementations (naive and blocked).
- `tests.h/cpp` – testing framework (correctness tests, timing benchmarks, optional reference comparisons).
- `CMakeLists.txt` – build configuration with support for Eigen, MKL, and RISC-V cross-compilation.
- Toolchain files – `x86_icpx2023.cmake` (Intel ICPX), `risk-V_gcc.cmake` (RISC-V GCC) and `risk-V_clang.cmake` (RISC-V CLANG).

### Example Output
```text
Use "--help" to see additional options.

TestSystem:
Testing with values type: double
Requires 0.119209Gb of RAM
Reference test library: Eigen 5.0.0

------------------------------------------- Test time -------------------------------------------
Testing with n = 2000, 2 times:

1) Infinite cond(A): 1.74396e-291; Matrix is good-conditioned. Test result: true. LU Time: 266
   Reference test 1. LU Time: 162. Test result: true

2) Infinite cond(A): 1.74396e-291; Matrix is good-conditioned. Test result: true. LU Time: 189
   Reference test 2. LU Time: 131. Test result: true

Minimum time for init random matrix: 35 ms
Minimum time for LU decomposition: 189 ms
Minimum time for reference LU decomposition: 131 ms
Minimum total time: 224 ms

Total test result: 100%
Correct count: 2
Incorrect count: 0
-------------------------------------------------------------------------------------------------
```
License
----------
ArchBenchs_LU_Decomposition is licensed under the MIT License. See [LICENSE](https://github.com/UuAcC/ArchBenchs/blob/LU_Decomposition/LICENSE) for details.

Authors
-------
### Developer
- [Simonov Maksim](https://github.com/UuAcC)
### Supervisors
- [Dr. Iosif Meyerov](https://sites.google.com/site/iosifmeyeroveng/)
- [Valentin Volokitin](https://github.com/ValentinV95)
- [Elena Panova](https://github.com/PanovaElena)
