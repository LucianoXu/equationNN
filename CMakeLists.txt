cmake_minimum_required(VERSION 3.10)
project(EQUATIONNN LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set the python executable
set(PYTHON_EXECUTABLE "/Users/xuyingte/anaconda3/bin/python3")
message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")

#######################################################
# Include FetchContent module

include(FetchContent)

# Download GoogleTest
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip  # Use the latest stable release
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Prevent GoogleTest from installing
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Declare pybind11 as an external project to fetch
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG master
)

# Make pybind11 available
FetchContent_MakeAvailable(pybind11)
#######################################################


####################################################
# Compiler Setting

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# Define build types with a default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build: Debug or Release" FORCE)
endif()

# Set the compiler
set(CMAKE_CXX_COMPILER "clang++")


# Set build flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O0 -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -DNDEBUG")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2 -g")
else()
    message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}")
endif()


# Enable testing
enable_testing()

####################################################
# C++

add_subdirectory(envbackend)
