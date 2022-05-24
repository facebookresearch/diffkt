find_package(OpenMP REQUIRED)
add_library(Sparse STATIC
  SparseFloatTensor.cpp
  SpMat.cpp
  Utils.cpp)
if (SPARSE_LIB STREQUAL "EIGEN")
  target_sources(Sparse PRIVATE
    ArithmeticEigen.cpp)
  target_link_libraries(Sparse Eigen3::Eigen)
elseif (SPARSE_LIB STREQUAL "MKL")
  target_sources(Sparse PRIVATE
    ArithmeticUtils.cpp
    ArithmeticMKL.cpp
    MKLCSR.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
  target_link_libraries(Sparse mkl_intel_lp64 mkl_intel_thread mkl_core)
else()
  target_sources(Sparse PRIVATE
    ArithmeticUtils.cpp
    ArithmeticOMP.cpp
    CSR.cpp)
endif()
target_link_libraries(Sparse OpenMP::OpenMP_CXX)
