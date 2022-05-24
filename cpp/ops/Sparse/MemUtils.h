/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_MEMUTILS_H_
#define OPS_MEMUTILS_H_

#include <stddef.h> // size_t
#include <initializer_list> // initializer_list
#include <vector> // vector
#include <utility> // swap
#include <string> // to_string
#include <type_traits> //std::is_same
#include "Sparse/DebugUtils.h" // Require

#ifdef MKL
#include <mkl.h> // mkl_calloc, mkl_malloc, mkl_free
#else
#include <stdlib.h> // malloc, calloc, free
#include <stdint.h> // int32_t
#endif

namespace ops {

#ifdef MKL
  /** Memory allocation interfaces
   * 128 is the alignment size */
  #define CALLOC(n, e) mkl_calloc(n, e, 128)
  #define MALLOC(s) mkl_malloc(s, 128)
  #define FREE(ptr) mkl_free(ptr)
#else
  /** Memory allocation interfaces */
  #define CALLOC(n, e) calloc(n, e)
  #define MALLOC(s) malloc(s)
  #define FREE(ptr) free(ptr)
#endif

  /** Basic types used */
  typedef int32_t INT;
  typedef float FLOAT;

  /** Types used in DimData, SparseFloatTensor and SpMat */
  typedef INT DimensionType;
  typedef FLOAT DataType;
  typedef INT OrdinalType;

  /** Memory allocation */
  template<typename T> inline
    void Calloc(T * & p, size_t n, size_t e) {
      p = (T *) CALLOC(n, e);
      Require(p != NULL, "Failed to allocate zero-initialized memory");
    }
  template<typename T> inline
    void Malloc(T * & p, size_t s) {
      p = (T *) MALLOC(s);
      Require(p != NULL, "Failed to allocate memory");
    }

  /**
   * A class to replace the usage of C++ vector in sparse computation.
   * Then
   *  - the malloc/free function can switch for different
   *    libraries, in particular, MKL prefers its allocation and free
   *    functions.
   *  - parallelism can be added in some operations such as assign
   *
   * Note : this class is similar with C++ vector but not the same.
   * This class only supports functions needed for sparse computation. Others,
   * such as copy constructor, copy assignment, appending, and automatic
   * resizing, are not supported. */
  template <class T_>
  class Array {
    private:
      /** The pointer to the memory */
      T_ * data_;
      /** The number of elements in the data_ array */
      size_t size_;

    public:
      /** A empty constructor */
      Array() : size_(0), data_(NULL) {}

      /** Build a Array with a given size */
      Array(size_t size) : size_(0), data_(NULL) { resize(size); }

      /** Build a Array with a given size and a initial value */
      Array(size_t size, T_ initialvalue) : size_(0), data_(NULL) {
        resize(size);
        assign(initialvalue);
      }

      /** Build a Array with a given data and size */
      template<typename T>
      Array(T * data, size_t size) : size_(0), data_(NULL) {
        assign(data, size);
      }

      /** Build a Array with a given vector */
      template<typename T>
      Array(std::vector<T> & vec) : size_(0), data_(NULL) {
        assign(vec.data(), vec.size());
      }

      /** Build a Array with a variable-length list of type T_ elements */
      Array( std::initializer_list<T_> list ) : size_(0), data_(NULL) {
        resize(list.size());
        size_t i=0;
        for (T_ it : list)
          data_[i++] = it;
      }

      /** Clean up the memory */
      ~Array() { clear(); }

      /** copy constructor */
      Array(const Array& other) = delete;
      /** copy assignment */
      Array& operator=(const Array& other) = delete;

      /** move constructor */
      Array(Array&& other) noexcept : size_(0), data_(NULL) {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
      }
      /** move assignment */
      Array& operator=(Array&& other) noexcept
      {
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
        other.clear();
        return *this;
      }

      /** Update the size of the data */
      // Note that: if size > size_ and size_ > 0, existing memory will be
      // released and new memory will be allocated.
      void resize(size_t size) {
        if (size == 0) clear();
        else if (size_ != size) {
          clear();
          Malloc(data_, size * sizeof(T_));
          size_ = size;
        }
      }

      /** Clean up the memory */
      void clear() {
        if (size_ != 0) {
          FREE(data_);
          size_ = 0;
          data_ = NULL;
        }
      }

      /** Assign the given value to data_ */
      void assign(T_ v) {
        #pragma omp parallel for
        for (size_t i=0; i<size_; i++) data_[i] = v;
      }

      /** Assign the given data to data_ */
      template<typename T>
      void assign(T * data, size_t size) {
        resize(size);
        #pragma omp parallel for
        for (size_t i=0; i<size_; i++) data_[i] = (T_)data[i];
      }

      /** Return the pointer */
      T_ * data() { return data_; }
      const T_ * data() const { return data_; }

      /** Return the number of elements in the array */
      size_t size() { return size_; }
      const size_t size() const { return size_; }

      /** Return the last element of the array */
      T_ back() {
        Require(size_ > 0, "Accessing the last element of an empty array.");
        return data_[size_ - 1];
      }

      /** Return the id-th element of the array */
      T_& operator[](const size_t id) {
        Require(id < size_, "Index (" + std::to_string(id) + ") should not exceed the size(" + std::to_string(size_) + ").");
        return data_[id];
      }
      const T_& operator[](const size_t id) const {
        Require(id < size_, "Index (" + std::to_string(id) + ") should not exceed the size(" + std::to_string(size_) + ").");
        return data_[id];
      }

      /** generate a vector that stores the same data */
      std::vector<T_> vector() {
        return std::vector<T_>(data_, data_ + size_);
      }
  };

  /** this operator is used to compare two objects of Array class of same
   * data type.
   * It's used in EXCEPT_EQ by gtest */
  template<class T>
  bool operator==( const Array<T> & l, const Array<T> & r) {
    if (l.size() != r.size()) return false;
    for (size_t i=0; i<l.size(); i++) if (l[i] != r[i]) return false;
    return true;
  }
} // namespace ops

#endif // OPS_MEMUTILS_H_
