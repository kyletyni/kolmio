# Specify the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Project name and languages
project(Kolmio LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories (for cl.hpp if manually downloaded)
include_directories(${PROJECT_SOURCE_DIR}/include)

# Find the OpenCL library
find_library(OpenCL_LIBRARY OpenCL)
if(NOT OpenCL_LIBRARY)
    message(FATAL_ERROR "OpenCL library not found!")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found!")
endif()

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(opencl_test main.cpp)

# Link the OpenCL and OpenCV libraries to the executable
target_link_libraries(opencl_test ${OpenCL_LIBRARY} ${OpenCV_LIBS})
