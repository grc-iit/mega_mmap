add_executable(gray-scott-upc
		main.cc
		gray-scott.cc
)

find_package(UPCXX REQUIRED)

target_link_libraries(gray-scott-upc MPI::MPI_C
		${HermesShm_LIBRARIES} yaml-cpp UPCXX::upcxx)
install(TARGETS gray-scott-upc
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX})