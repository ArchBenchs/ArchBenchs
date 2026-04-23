set(CMAKE_C_COMPILER riscv64-unknown-linux-gnu-clang)
set(CMAKE_CXX_COMPILER riscv64-unknown-linux-gnu-clang++)

set(CMAKE_CXX_FLAGS "-std=c++17 -O2 -g -fopenmp -march=rv64gcv_zvfh -mabi=lp64d -DBLOCK_SIZE=32")
set(CMAKE_EXE_LINKER_FLAGS "-fopenmp -lomp -static")