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

if(DEFINED DO_REFERENCE_TEST)
	target_compile_definitions(${project_name} PRIVATE DO_REFERENCE_TEST=${DO_REFERENCE_TEST})
	target_include_directories(${project_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/eigen-3.4.0")
endif()

