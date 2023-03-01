cmake_minimum_required (VERSION 3.10)

#------------------------------------------------------------------------------
# Compiler optimization
#------------------------------------------------------------------------------
message(${CMAKE_BUILD_TYPE})
if (CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  message("This IS a release build")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
  message("This is NOT a release build")
endif()

#------------------------------------------------------------------------------
# Compiler optimization
#------------------------------------------------------------------------------
add_executable(hermes_userfault hermes_userfault.cc)