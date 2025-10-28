cmake_minimum_required(VERSION 3.8)

set(project_name "Matrix_Project")

project(${project_name})

file(GLOB hdrs "source/*.h*")
file(GLOB srcs "source/*.cpp")

add_executable(${project_name} ${srcs} ${hdrs})
