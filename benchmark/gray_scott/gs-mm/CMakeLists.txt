add_executable(gray-scott-mm
		main.cc
		gray-scott.cc
)

target_link_libraries(gray-scott-mm MPI::MPI_C
		${Hermes_LIBRARIES} ${HermesShm_LIBRARIES} yaml-cpp)
install(TARGETS gray-scott-mm
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})