cmake_minimum_required(VERSION 3.3 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

# -fPIC is needed to add STATIC lib on Linux
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3")
project(diffkt-ops)

# JNI
set(JAVA_INCLUDE_PATH $ENV{JAVA_INCLUDE_PATH})
set(JAVA_INCLUDE_PATH2 $ENV{JAVA_INCLUDE_PATH2})
set(JAVA_AWT_INCLUDE_PATH $ENV{JAVA_INCLUDE_PATH})

find_package(JNI REQUIRED)
include_directories("${JNI_INCLUDE_DIRS}")

#EIGEN https://eigen.tuxfamily.org/dox/TopicCMakeGuide.html
find_package (Eigen3 3.3 NO_MODULE)

# MKL
set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)

# For sparse computation,
# let the user to choose MKL or Eigen or OMP, or None
# MKL and Eigen are third-party libraries, and they need to be installed to use.
# OMP is our OpenMP implementation.
# None means user doesn't prefer any, and then the sparse computation is selected
# by predefined preference which are: MKL > Eigen > OMP
set(SPARSE_LIB "NONE" CACHE STRING "Choose the sparse lib to use")
set(SPARSE_LIBS "MKL" "EIGEN" "OMP" "NONE")
set_property(CACHE SPARSE_LIB PROPERTY STRINGS SPARSE_LIBS)
if (NOT SPARSE_LIB IN_LIST SPARSE_LIBS)
  message(FATAL_ERROR "SPARSE_LIB is set as ${SPARSE_LIB}. Please choose SPARSE_LIB among ${SPARSE_LIBS}")
endif()

if (SPARSE_LIB STREQUAL "MKL" AND NOT BLAS_FOUND)
  message(FATAL_ERROR "Cannot find MKL, but forced to use it.
  If you installed MKL already, related environment variables might not be set successfully.
  You can solve that by running:
      source /opt/intel/oneapi/setvars.sh")
endif()
if (SPARSE_LIB STREQUAL "EIGEN" AND NOT Eigen3_FOUND)
  message(FATAL_ERROR "Cannot find Eigen, but forced to use it")
endif()
if (SPARSE_LIB STREQUAL "NONE")
  if (BLAS_FOUND)
    set(SPARSE_LIB "MKL")
  elseif (Eigen3_FOUND)
    set(SPARSE_LIB "EIGEN")
  else()
    set(SPARSE_LIB "OMP")
  endif()
endif()

if (SPARSE_LIB STREQUAL "MKL")
  if(NOT DEFINED ENV{MKLROOT} OR NOT DEFINED ENV{CMPLR_ROOT})
    message(FATAL_ERROR "Required envirinment variables: MKLROOT and CMPLR_ROOT, are not set.
    If MKL is installed correctly, you can solve this by running:
      source /opt/intel/oneapi/setvars.sh")
  endif()
  set(CMAKE_BUILD_RPATH "$ENV{MKLROOT}/lib/;$ENV{CMPLR_ROOT}/mac/compiler/lib/")
  add_compile_options(-DMKL)
  message(STATUS "Using MKL for sparse computation")
elseif (SPARSE_LIB STREQUAL "EIGEN")
  add_compile_options(-DEIGEN)
  message(STATUS "Using Eigen for sparse computation")
else()
  message(STATUS "Using our OpenMP implementation for sparse computation")
endif()

# DNNL https://intel.github.io/mkl-dnn/dev_guide_transition_to_dnnl.html
find_package(dnnl CONFIG REQUIRED
  HINTS ${CMAKE_CURRENT_SOURCE_DIR}/dnnl/lib/cmake/dnnl)

# OpenMP
find_package(OpenMP REQUIRED)

### GoogleTest ###

# Source: https://github.com/google/googletest/tree/master/googletest#incorporating-into-an-existing-cmake-project
# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.

### Our targets ###

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

# We want to output the library in the tensor API's resources directory for publishing purposes
set(RESOURCES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../kotlin/api/src/main/resources)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${RESOURCES_PATH})

# The Ops library setup is based on Step 2 of the CMake tutorial
# https://cmake.org/cmake/help/latest/guide/tutorial/index.html#adding-a-library-step-2
add_subdirectory(Dnnl)
add_subdirectory(Sparse)
add_subdirectory(Math)
add_subdirectory(Predicate)

# The JNI shared libraries
add_library(ops_jni SHARED ops_helper.h ops_helper.cpp)
add_library(sparseops_jni SHARED SparseOps.h SparseOps.cpp)
add_library(dnnlops_jni SHARED DnnlOps.h DnnlOps.cpp)

# PRIVATE means used in implementation and not API
# https://cmake.org/pipermail/cmake/2016-May/063400.html
target_link_libraries(sparseops_jni PRIVATE Sparse)
target_link_libraries(dnnlops_jni PRIVATE DNNL::dnnl Dnnl)
target_link_libraries(ops_jni PRIVATE Math PRIVATE Predicate PRIVATE)

# Include Java includes as per https://stackoverflow.com/a/37889423
target_include_directories(ops_jni PUBLIC $ENV{JAVA_HOME}/include)
target_include_directories(dnnlops_jni PUBLIC $ENV{JAVA_HOME}/include)
target_include_directories(sparseops_jni PUBLIC $ENV{JAVA_HOME}/include)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    target_include_directories(ops_jni PUBLIC $ENV{JAVA_HOME}/include/darwin)
    target_include_directories(dnnlops_jni PUBLIC $ENV{JAVA_HOME}/include/darwin)
    target_include_directories(sparseops_jni PUBLIC $ENV{JAVA_HOME}/include/darwin)
endif()

# Testing
enable_testing()
add_subdirectory(test)
