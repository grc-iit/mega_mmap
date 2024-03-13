cmake_minimum_required (VERSION 3.10)

# add_executable(mm_sort mm_sort.cc)
# target_link_libraries(mm_sort ${Hermes_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)

add_executable(mm_hermes_test hermes_test.cc
        ../include/mega_mmap/vector.h
)
target_link_libraries(mm_hermes_test ${Hermes_LIBRARIES})

add_executable(mm_kmeans mm_kmeans.cc)
target_link_libraries(mm_kmeans ${Hermes_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)

add_executable(mm_kmeans_df mm_kmeans_df.cc)
target_link_libraries(mm_kmeans_df ${Hermes_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)

add_executable(mm_random_forest mm_random_forest.cc)
target_link_libraries(mm_random_forest ${Hermes_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)

add_executable(mm_random_forest_df mm_random_forest_df.cc)
target_link_libraries(mm_random_forest_df ${Hermes_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)
#
#add_executable(mm_dbscan mm_dbscan.cc
#        ../include/mega_mmap/pattern/seq_iter.h)
#target_link_libraries(mm_dbscan ${Hermes_LIBRARIES} MPI::MPI_CXX arrow_shared parquet_shared)
#
#add_executable(mm_gadget2conv mm_gadget2conv.cc)
#target_link_libraries(mm_gadget2conv ${Hermes_LIBRARIES}
#        MPI::MPI_CXX arrow_shared parquet_shared HDF5::HDF5)

install(TARGETS mm_hermes_test mm_kmeans mm_kmeans_df mm_random_forest mm_random_forest_df # mm_dbscan mm_gadget2conv
        RUNTIME DESTINATION bin)

install(FILES pandas_kmeans.py pandas_random_forest.py pandas_dbscan.py
        spark_kmeans.py spark_random_forest.py
        DESTINATION scripts)