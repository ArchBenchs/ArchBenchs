#!/bin/bash -x
cmake -S . -B build -DCMAKE_CXX_COMPILER=g++
cmake --build build --config Debug
