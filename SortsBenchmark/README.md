ArchBenchs_Quick_sort
============

![C++17](https://img.shields.io/badge/C%2B%2B-17-red.svg)
![Platforms](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20riscv-lightgrey.svg)
![OpenMP](https://img.shields.io/badge/OpenMP-supported-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

The project ArchBenchs_Quick_sort is a part of ArchBenchs benchmarks and provides a high-performance implementation of quick sort.

Getting Started
---------------
This repository doesn't use git submodules. To clone and use the "SortsBenchmark" directory in the repository:
```bash
git clone https://github.com/ArchBenchs/ArchBenchs.git
```

Installing
----------

### Dependencies
- `C++17` compatible compiler (GCC (for RISC-V), Intel ICPX (for x86))
- `OpenMP` (for parallelization)

### Build
Compile the file that uses the sorting implementation (included quick_sort.cpp). Available toolchain files, which are used in development:
```bash
icpx -g -qopenmp -O2 -std=c++17 your_file.cpp -o run
```
```bash
riscv64-unknown-linux-gnu-g++ -std=c++17 -O2 -g -fopenmp -march=rv64gcv_zvfh -mabi=lp64d your_file.cpp -o run
```

Documentation
----------
### Algorithm Overview
Use the parallel_quick_sort() function, which is located in the file quick_sort.cpp


License
----------
ArchBenchs_Quick_sort is licensed under the MIT License. See [LICENSE](https://github.com/ArchBenchs/ArchBenchs/blob/main/LICENSE) for details.

Authors
-------
### Developer
- [Eduard Brusnigin](https://github.com/EduardBrusnigin)
### Supervisors
- [Iosif Meyerov](https://sites.google.com/site/iosifmeyeroveng/)
- [Valentin Volokitin](https://github.com/ValentinV95)
- [Elena Panova](https://github.com/PanovaElena)
