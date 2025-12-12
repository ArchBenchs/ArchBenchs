add_library(${PROG_LIBRARY} STATIC 
	Class_Kernel.cpp
	Functions.cpp

)

target_link_libraries(${PROG_LIBRARY} ${OpenCV_LIBS})