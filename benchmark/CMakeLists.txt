cmake_minimum_required (VERSION 3.10)

add_executable(sort parallel_sort.cc)
target_link_libraries(sort ${HermesShm_LIBRARIES} MPI::MPI_CXX)

add_executable(kmeans_df kmeans_df.cc)
target_link_libraries(kmeans_df ${HermesShm_LIBRARIES} MPI::MPI_CXX)