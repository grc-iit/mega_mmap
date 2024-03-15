add_executable(gray-scott-mm
		${CMAKE_CURRENT_SOURCE_DIR}/main.cc
		${CMAKE_CURRENT_SOURCE_DIR}/gray-scott.cc
)

target_link_libraries(gray-scott-mm
		${Hermes_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)
install(TARGETS gray-scott-mm
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})