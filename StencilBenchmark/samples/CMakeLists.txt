add_executable(sample_program
	sample.cpp

)

target_link_libraries(sample_program PRIVATE 
	${PROG_LIBRARY} 
	${OpenCV_LIBS}
)

find_path(OpenCV_DLL_DIR
	NAMES opencv_world4120d.dll opencv_world4120.dll
	PATHS "${OpenCV_DIR}/x64/vc16/bin"
	NO_DEFAULT_PATH
)

if(OpenCV_DLL_DIR)
	file(GLOB OpenCV_DLLs "${OpenCV_DLL_DIR}/*.dll")
	foreach(dll ${OpenCV_DLLs})
		add_custom_command(TARGET sample_program POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			"${dll}"
			$<TARGET_FILE_DIR:sample_program>
			COMMENT "Copying OpenCV DLLs"
		)
	endforeach()
endif()