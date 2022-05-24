if (SPARSE_LIB STREQUAL "MKL")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
endif()

add_library(SparseTestUtils STATIC
  SparseTestUtils.cpp)
target_link_libraries(SparseTestUtils gtest_main Sparse)

add_executable(SparseArithmeticTest SparseArithmeticTest.cpp)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
target_link_libraries(SparseArithmeticTest PUBLIC SparseTestUtils -liomp5)
else()
target_link_libraries(SparseArithmeticTest PUBLIC SparseTestUtils)
endif()
add_test(NAME SparseArithmeticTest COMMAND SparseArithmeticTest)

add_executable(SparseArithmeticPerfTest SparseArithmeticPerfTest.cpp)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
target_link_libraries(SparseArithmeticPerfTest PUBLIC SparseTestUtils -liomp5)
else()
target_link_libraries(SparseArithmeticPerfTest PUBLIC SparseTestUtils)
endif()

add_executable(ArithmeticDnnlTest ArithmeticDnnlTest.cpp)
target_link_libraries(ArithmeticDnnlTest
                      PUBLIC
                        gtest_main
                        Dnnl)
add_test(NAME ArithmeticDnnlTest COMMAND ArithmeticDnnlTest)

add_executable(BatchNormTest BatchNormTest.cpp)
target_link_libraries(BatchNormTest
                      PUBLIC
                        gtest_main
                        Dnnl)
add_test(NAME BatchNormTest COMMAND BatchNormTest)

add_executable(ConvTest ConvTest.cpp)
target_link_libraries(ConvTest
                      PUBLIC
                        gtest_main
                        Dnnl)
add_test(NAME ConvTest COMMAND ConvTest)

add_executable(PoolingTest PoolingTest.cpp)
target_link_libraries(PoolingTest
                      PUBLIC
                        gtest_main
                        Dnnl)
add_test(NAME PoolingTest COMMAND PoolingTest)

add_executable(ReduceTest ReduceTest.cpp)
target_link_libraries(ReduceTest
                      PUBLIC
                        gtest_main
                        Dnnl)
add_test(NAME ReduceTest COMMAND ReduceTest)

add_executable(ReluTest ReluTest.cpp)
target_link_libraries(ReluTest
                      PUBLIC
                        gtest_main
                        Dnnl)
add_test(NAME ReluTest COMMAND ReluTest)
