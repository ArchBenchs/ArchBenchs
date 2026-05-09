add_executable(sample_program
	sample.cpp

)

target_link_libraries(sample_program PRIVATE 
	${PROG_LIBRARY} 
)