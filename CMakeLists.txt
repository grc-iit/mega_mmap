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
  add_compile_definitions(HERMES_LOG_VERBOSITY=10)
else()
  message("IN RELEASE MODE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3")
  add_compile_definitions(HERMES_LOG_VERBOSITY=1)
endif()
add_compile_options(-march=native -fomit-frame-pointer)

#------------------------------------------------------------------------------
# Find packages
#------------------------------------------------------------------------------
# Hermes
#find_package(Hermes CONFIG REQUIRED)
#message(STATUS "found Hermes at ${Hermes_DIR}")

# Apache Arrow
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# HermesShm
find_package(HermesShm CONFIG REQUIRED)
message(STATUS "found hermes_shm.h at ${HermesShm_INCLUDE_DIRS}")
include_directories(${HermesShm_INCLUDE_DIRS})

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

#------------------------------------------------------------------------------
# Includes
#------------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

#------------------------------------------------------------------------------
# Test
#------------------------------------------------------------------------------
add_subdirectory(test/unit)
add_subdirectory(benchmark)