set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

set(CMAKE_CXX_FLAGS "-qopenmp -g -O2 -march=icelake-server -std=c++17 ${CMAKE_CXX_FLAGS}")

if(DEFINED REFERENCE_TEST)
	if (${REFERENCE_TEST} STREQUAL "mkl")
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qmkl=parallel")
	endif()
endif()