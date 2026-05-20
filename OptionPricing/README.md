ArchBenchs_Options_Pricing
============

![C++17](https://img.shields.io/badge/C%2B%2B-17-red.svg)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20riscv-lightgrey.svg)
![OpenMP](https://img.shields.io/badge/OpenMP-supported-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

The project ArchBenchs_Options_Pricing is a part of ArchBenchs benchmarks and provides a high-performance implementation of several European and Bermudan options pricing algorithms.

Getting Started
---------------
This repository uses the OpenRand git submodule. To clone and use the "OptionPricing" directory in the repository:
```bash
git clone https://github.com/ArchBenchs/ArchBenchs.git
cd ArchBenchs
git submodule update --init --recursive
```

Installing
----------

### Dependencies
- `C++17` compatible compiler (GCC or Clang (for x86 or for RISC-V), Intel ICPX)
- [`CMake`](https://cmake.org/) 3.8 or higher
- `OpenMP` (for parallelization)
- `OpenRand` (in a submodule, for the random number generator)
- (Optional) [`Intel MKL`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) -- there's a version that uses MKL for the random number generator

### Build
Use CMake to build project. Available toolchain files, which are used in development:
```bash
CC=gcc CXX=g++ cmake ..
```
```bash
CC=riscv64-unknown-linux-gnu-gcc CXX=riscv64-unknown-linux-gnu-g++ cmake -DCMAKE_SYSTEM_PROCESSOR=riscv64 -DRISCV=ON ..
```

Documentation
----------
### Algorithm Overview
This is an implementation of these algorithms:
 - Monte Carlo, for numerically calculating the price of European call-max options. Also included is a version, where the price of different stocks has non-zero corellation.
 - Broadie Glasserman Random Trees, for Bermudan options
 - Broadie Glasserman Stochastic Mesh, a better scaling algorithm for Bermudan options.


License
----------
ArchBenchs_Quick_sort is licensed under the MIT License. See [LICENSE](https://github.com/ArchBenchs/ArchBenchs/blob/main/LICENSE) for details.

Credits 
-------
 - Broadie Glasserman Random Trees (M. Broadie; and P. Glasserman: Pricing American-style securities using simulation. Journal of Economic Dynamics and Control, 21 (1997): 1323-1352.)
 - Broadie Glasserman Stochastic Mesh (M. Broadie, and P. Glasserman: A stochastic mesh method for pricing high-dimensional American options. working paper, Columbia Business School. Columbia University, 1997)
 - [toms462](https://people.sc.fsu.edu/~jburkardt/cpp_src/toms462/toms462.html) -- used in a multivariate analytical European option func

Authors
-------
### Developer
- [Ermolev Tihon](https://github.com/Tixorus)
### Supervisors
- [Iosif Meyerov](https://sites.google.com/site/iosifmeyeroveng/)
- [Valentin Volokitin](https://github.com/ValentinV95)
- [Elena Panova](https://github.com/PanovaElena)
