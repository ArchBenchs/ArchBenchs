set(CMAKE_C_COMPILER "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe")
set(CMAKE_CXX_COMPILER "C:/Program Files (x86)/Intel/oneAPI/compiler/latest/bin/icx.exe")

# Спроси, что это все такое
set(INTEL_OPTIMIZATION_FLAGS "/O2 /QxHost")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${INTEL_OPTIMIZATION_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${INTEL_OPTIMIZATION_FLAGS}")

set(CMAKE_CROSSCOMPILING TRUE)

set(CMAKE_C_COMPILER_AR "llvm-ar")
set(CMAKE_C_COMPILER_RANLIB "llvm-ranlib")
set(CMAKE_CXX_COMPILER_AR "llvm-ar")
set(CMAKE_CXX_COMPILER_RANLIB "llvm-ranlib")