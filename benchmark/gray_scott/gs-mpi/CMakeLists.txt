add_executable(gray-scott-mpi
		main.cc
		gray-scott.cc
)
target_link_libraries(gray-scott-mpi MPI::MPI_C ${HermesShm_LIBRARIES} yaml-cpp)
install(TARGETS gray-scott-mpi
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})