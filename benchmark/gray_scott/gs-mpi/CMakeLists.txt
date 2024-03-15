add_executable(gray-scott-mpi
		${CMAKE_CURRENT_SOURCE_DIR}/main.cc
		${CMAKE_CURRENT_SOURCE_DIR}/gray-scott.cc
)
target_link_libraries(gray-scott-mpi
		MPI::MPI_C
		${HermesShm_LIBRARIES} yaml-cpp)
install(TARGETS gray-scott-mpi
		RUNTIME DESTINATION bin)