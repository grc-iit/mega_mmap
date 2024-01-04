cmake_minimum_required (VERSION 3.10)

add_executable(mm_sort mm_sort.cc)
target_link_libraries(mm_sort ${HermesShm_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)

add_executable(mm_kmeans mm_kmeans.cc)
target_link_libraries(mm_kmeans ${HermesShm_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)

add_executable(mm_kmeans_df mm_kmeans_df.cc)
target_link_libraries(mm_kmeans_df ${HermesShm_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)

add_executable(mm_random_forest mm_random_forest.cc)
target_link_libraries(mm_random_forest ${HermesShm_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)

add_executable(mm_random_forest_df mm_random_forest_df.cc)
target_link_libraries(mm_random_forest_df ${HermesShm_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)
