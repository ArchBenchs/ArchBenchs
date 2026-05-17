ArchBenchs
=======

![C++17](https://img.shields.io/badge/C%2B%2B-17-red.svg)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20riscv-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Description
----------
The ArchBenchs project is a collection of open-source portable benchmarks that can run on both x86 and RISC-V processors. The project's primary goal is to create portable educational benchmarks for use as illustrative material for the textbook "High-Performance Parallel Computing. 100 Assignments for Extended Lab Workshop." ISBN 978-5-9221-1816-3. Currently, the following benchmarks have been implemented:
- [`LU Decomposition`](DecompositionLU) - implementation of the block LU decomposition algorithm using OpenMP;
- [`Quick sort`](SortsBenchmark) - implementation of a parallel version of quick sort with optimizations;
- [`Option Pricing`](OptionPricing) - implementation of a variety of numerical option pricing algorithms, with a focus on the effeciency of parallelization for various architectures;
- [`Numerical integration`](IntegralBenchmark) - implementation of parallel vectorized version of rectangle, trapezoid, and Monte Carlo integration for RISC-V;
- [`Stencil`](StencilBenchmark) - implementation of a parallel vectorized version of matrix convolution;
- [`LZ77 archivation`](ArchiveBenchmark) - implementation of parallel version of LZ77 archivation algorithm based on suffix array.
This set of benchmarks will be expanded. Detailed information about each benchmark is described in its README file.

License
----------
ArchBenchs_LU_Decomposition is licensed under the MIT License. See [LICENSE](https://github.com/UuAcC/ArchBenchs/blob/LU_Decomposition/LICENSE) for details.

Supervisors
-------
- [Dr. Iosif Meyerov](https://sites.google.com/site/iosifmeyeroveng/)
- [Valentin Volokitin](https://github.com/ValentinV95)
- [Elena Panova](https://github.com/PanovaElena)
