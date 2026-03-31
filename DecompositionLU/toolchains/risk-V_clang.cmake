set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_CXX_FLAGS "-std=c++17 -O2 -g -fopenmp -march=rv64gcv_zvfh -mabi=lp64d")
set(CMAKE_EXE_LINKER_FLAGS "-fopenmp")