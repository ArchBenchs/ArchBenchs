set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

set(CMAKE_CXX_FLAGS "-qopenmp -g -O2 -march=icelake-server -std=c++17 ${CMAKE_CXX_FLAGS}")