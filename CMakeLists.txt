cmake_minimum_required (VERSION 3.10)
project(MegaMmap)

#-----------------------------------------------------------------------------
# Compiler Optimization
#-----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("IN DEBUG MODE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")
  add_compile_definitions(HERMES_LOG_VERBOSITY=1)
else()
  message("IN RELEASE MODE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3")
  add_compile_definitions(HERMES_LOG_VERBOSITY=1)
endif()
add_compile_options(-march=native -fomit-frame-pointer)

#------------------------------------------------------------------------------
# Setup CMake Environment
#------------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables.")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY
        ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries.")

#------------------------------------------------------------------------------
# Setup install and output Directories
#------------------------------------------------------------------------------
if(NOT MEGAMMAP_INSTALL_BIN_DIR)
  set(MEGAMMAP_INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin)
endif()
if(NOT MEGAMMAP_INSTALL_LIB_DIR)
  set(MEGAMMAP_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
endif()
if(NOT MEGAMMAP_INSTALL_INCLUDE_DIR)
  set(MEGAMMAP_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
endif()
if(NOT MEGAMMAP_INSTALL_DATA_DIR)
  set(MEGAMMAP_INSTALL_DATA_DIR ${CMAKE_INSTALL_PREFIX}/share)
endif()

#------------------------------------------------------------------------------
# Find packages
#------------------------------------------------------------------------------
# This is for compatability with SPACK
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# HermesShm
find_package(HermesShm CONFIG REQUIRED)
message(STATUS "found hermes_shm.h at ${HermesShm_INCLUDE_DIRS}")
include_directories(${HermesShm_INCLUDE_DIRS})
link_directories(${HermesShm_LIBRARY_DIRS})

# Hermes
find_package(Hermes CONFIG REQUIRED)
message(STATUS "found Hermes at ${Hermes_DIR}")
include_directories(${Hermes_INCLUDE_DIRS})
link_directories(${Hermes_LIBRARY_DIRS})
message("LIBRARIES: ${Hermes_LIBRARIES}")
message("INCLUDE: ${Hermes_INCLUDE_DIRS}")

# Apache Arrow
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# YAML-CPP
find_package(yaml-cpp REQUIRED)
message(STATUS "found yaml-cpp at ${yaml-cpp_DIR}")

# Catch2
find_package(Catch2 3.0.1 REQUIRED)
message(STATUS "found catch2.h at ${Catch2_CXX_INCLUDE_DIRS}")

# MPICH
find_package(MPI REQUIRED COMPONENTS C CXX)
message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")

# OpenMP
find_package(OpenMP REQUIRED COMPONENTS C CXX)
message(STATUS "found omp.h at ${OpenMP_CXX_INCLUDE_DIRS}")

# Cereal
find_package(cereal REQUIRED)
if(cereal)
  message(STATUS "found cereal")
endif()

# Pkg-Config
find_package(PkgConfig REQUIRED)
if(PkgConfig)
  message(STATUS "found pkg config")
endif()

# HDF5
set(MEGAMMAP_REQUIRED_HDF5_VERSION 1.14.0)
find_package(HDF5 ${MEGAMMAP_REQUIRED_HDF5_VERSION} REQUIRED)


#------------------------------------------------------------------------------
# Includes
#------------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------
function(jarvis_test test_name)
  add_custom_target(${test_name}
          COMMAND jarvis pipeline run yaml "${CMAKE_SOURCE_DIR}/test/unit/pipelines/${test_name}.yaml")
endfunction()
add_subdirectory(test/unit)
add_subdirectory(benchmark)