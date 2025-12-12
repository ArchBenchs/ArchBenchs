cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME STENCIL)
project(${PROJECT_NAME})

set(OpenCV_DIR "C:/Users/IVAN/opencv/build" CACHE PATH "" FORCE)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROG_LIBRARY "${PROJECT_NAME}")

set(PROG_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

include_directories(${PROG_INCLUDE})

add_subdirectory(src)
add_subdirectory(samples)