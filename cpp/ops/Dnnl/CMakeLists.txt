add_library(Dnnl STATIC
  ArithmeticDnnl.cpp
  BatchNorm.cpp
  Conv.cpp
  LogSoftmax.cpp
  Pooling.cpp
  Reduce.cpp
  Relu.cpp
  Utils.cpp)

target_link_libraries(Dnnl DNNL::dnnl OpenMP::OpenMP_CXX )
