cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(diffkt-gpu-ops)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# libtorch
find_package(Torch REQUIRED
                HINTS ${CMAKE_CURRENT_SOURCE_DIR}/libtorch)

# JNI
set(JAVA_INCLUDE_PATH $ENV{JAVA_INCLUDE_PATH})
find_package(JNI REQUIRED)
include_directories("${JNI_INCLUDE_DIRS}")

# We want to output the library in the tensor API's resources directory for publishing purposes
set(RESOURCES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../kotlin/api/src/main/resources)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${RESOURCES_PATH})

add_executable(example-app example-app.cpp)
target_link_libraries(example-app torch)
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

add_library(gpuops_jni SHARED ops_jni.h ops_jni.cpp)
# PRIVATE means used in implementation and not API
# https://cmake.org/pipermail/cmake/2016-May/063400.html
target_link_libraries(gpuops_jni PRIVATE torch)
