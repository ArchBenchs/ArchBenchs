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

if(DEFINED REFERENCE_TEST)
    target_compile_definitions(${project_name} PRIVATE REFERENCE_TEST=${REFERENCE_TEST})
    if(${REFERENCE_TEST} STREQUAL "eigen")
        message(STATUS "Using Eigen as reference")
        target_include_directories(${project_name} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/eigen-5.0.0")
    elseif(${REFERENCE_TEST} STREQUAL "mkl")
	set(MKL_THREADING "openmp" CACHE STRING "MKL threading model")
	set(MKL_INTERFACE "lp64" CACHE STRING "MKL interface")
	set(MKL_LINK "dynamic" CACHE STRING "MKL link type")
        message(STATUS "Using MKL as reference")
        find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
        target_link_libraries(${project_name} PRIVATE MKL::MKL)
    endif()
endif()

