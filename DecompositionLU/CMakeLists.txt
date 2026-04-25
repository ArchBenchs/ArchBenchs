cmake_minimum_required(VERSION 3.8)

set(project_name "LU_Benchmark")

project(${project_name})

file(GLOB hdrs "source/*.h*")
file(GLOB srcs "source/*.cpp")

add_executable(${project_name} ${srcs} ${hdrs})

if(DEFINED PRINT_BLOCK_TIMES)
    target_compile_definitions(${project_name} PRIVATE PRINT_BLOCK_TIMES=${PRINT_BLOCK_TIMES})
endif()

if(DEFINED BLOCK_SIZE)
    target_compile_definitions(${project_name} PRIVATE BLOCK_SIZE=${BLOCK_SIZE})
endif()


