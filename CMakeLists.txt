cmake_minimum_required(VERSION 3.10)
project(EQUATIONNN LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


#######################################################
# Include FetchContent module

include(FetchContent)

# GoogleTest
# Download GoogleTest
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip  # Use the latest stable release
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Prevent GoogleTest from installing
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


# Python
# if PYTHON_EXECUTABLE is not set, try to find python with find_package
if (NOT DEFINED PYTHON_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
else()
    # Check the python executable
    if(NOT EXISTS ${PYTHON_EXECUTABLE})
        message(FATAL_ERROR "Python executable not found: ${PYTHON_EXECUTABLE}")
    endif()
endif()
message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")


# ANTLR4
# set the ANTLR4 executable
set(ANTLR4 "${CMAKE_CURRENT_SOURCE_DIR}/extern/antlr-4.13.2-complete.jar")
message(STATUS "ANTLR4 executable: ${ANTLR4}")



# pybind11
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
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define build types with a default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build: Debug or Release" FORCE)
endif()

# Set the compiler to clang++
# check whether clang++ is available
find_program(CLANGPP clang++)
if(CLANGPP)
    set(CMAKE_CXX_COMPILER ${CLANGPP})
    message(STATUS "Using clang++ compiler: ${CLANGPP}")
else()
    message(FATAL_ERROR "clang++ compiler not found")
endif()


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
