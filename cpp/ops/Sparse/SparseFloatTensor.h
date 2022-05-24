/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef OPS_SPARSEFLOATTENSOR_H_
#define OPS_SPARSEFLOATTENSOR_H_

#include "DebugUtils.h"
#include "SpMat.h"

#include <algorithm>
#include <vector>

namespace ops {

  /** This stores the sparse structure */
  class DimData {
    private:
      /** stores indices for the non-zero/non-empty
       * inner_.size() should be equal to outer_.back() */
      Array<DimensionType> inner_;
      /** stores starting positions in inner_ for each tensor
       * outer_ should be sorted increasingly and starts with 0 */
      Array<OrdinalType> outer_;

    public:
      /** empty constructor */
      DimData() {}
      /** constructor from inner and outer */
      DimData(Array<DimensionType> inner, Array<OrdinalType> outer) :
          inner_(std::move(inner)), outer_(std::move(outer)) {}
      /** copy constructor */
      DimData(const DimData& other) = delete;
      /** copy assignment */
      DimData& operator=(const DimData& other) = delete;
      /** move constructor */
      DimData(DimData&& other) noexcept
        : inner_(std::move(other.inner_)), outer_(std::move(other.outer_)) {}
      /** move assignment */
      DimData& operator=(DimData&& other) noexcept
      {
        inner_ = std::move(other.inner_);
        outer_ = std::move(other.outer_);
        return *this;
      }

      /** get the reference for inner_ */
      Array<DimensionType> & inner() { return inner_; }
      /** get a const reference for inner_ */
      const Array<DimensionType> & inner() const { return inner_; }
      /** get the reference for outer_ */
      Array<OrdinalType> & outer() { return outer_; }
      /** get a const reference for outer_ */
      const Array<OrdinalType> & outer() const { return outer_; }

      #ifdef DEBUG
      /** check whether if the data is correct */
      void checkDataCorrectness() {
        if (outer_.size() == 0)
          Require ( inner_.size() == 0, "In the DimData, when the outer is empty, the inner array should be empty as well");
        else {
          Require ( outer_[0] == 0, "In the DimData, the first element of the outer should be zero" );
          Require ( outer_.back() == inner_.size(), "In the DimData, the outer's last element should represent the size of the inner" );
          for (size_t i=1; i<outer_.size(); i++) Require (outer_[i-1] <= outer_[i], "In the DimData, data in the outer should be sorted increasingly.");
        }
      }
      #endif // DEBUG
  };

  /**
   * This memory wrapper holds pointers to the memory in [mem_] and will
   * free() them when it is destroyed.
   *
   * This is useful when another object [T] depends on some memory
   * external to it, and that memory may either, at runtime, (1) have
   * something that will free it later or (2) not have any other object
   * that can free it.
   *
   * In this case, MemWrapper<[T]> can be used.  If it is constructed with
   * pointers, it will free those, otherwise it will do nothing.  [T] can
   * be accessed through get(). */
  template <class T>
  class MemWrapper {
    private:
      /** Any memory that needs to be freed */
      std::vector<void*> mem_;

      /** The constructed object */
      T obj_;
    public:
      /** Build the memory wrapper with the given object and a set of
       * memory to free */
      MemWrapper(T obj, std::vector<void*> mem) :
        obj_(std::move(obj)), mem_(mem) {}
      /** Clean up the memory wrapper */
      ~MemWrapper() {
        for (void* mem : mem_) {
            FREE(mem);
        }
      }
      /** copy constructor */
      MemWrapper(const MemWrapper& other) = delete;
      /** copy assignment */
      MemWrapper& operator=(const MemWrapper& other) = delete;

      /** move constructor */
      MemWrapper(MemWrapper&& other) noexcept
        : obj_(std::move(other.obj_)) {
          std::swap(mem_, other.mem_);
      }
      /** move assignment */
      MemWrapper& operator=(MemWrapper&& other) noexcept
      {
        obj_ = std::move(other.obj_);
        std::swap(mem_, other.mem_);
        return *this;
      }

      /** Return the underlying object
       * Note: this is only safe while this class exists
       */
      T& get() { return obj_; }
      const T& get() const { return obj_; }
  };

  /** Holds a sparse tensor with internal DataType values
   * This class maps to the Java org/diffkt/SparseFloatTensor class
   *
   * The naming for fields is the same among this two classes, except each
   * field's name is ended with "_" here. That is:
   * SparseFloatTensor.shape_ -> org/diffkt/SparseFloatTensor.shape
   * SparseFloatTensor.values_ -> org/diffkt/SparseFloatTensor.values
   * SparseFloatTensor.dims_ -> org/diffkt/SparseFloatTensor.dims */
  class SparseFloatTensor {
    private:
      /** Holds the shape information for the tensor */
      Array<DimensionType> shape_;
      /** Holds the actual values of the tensor */
      Array<DataType> values_;
      /** Contains the sparsity structure information for the tensor */
      std::vector<DimData> dims_;
    public:
      /** Empty constructor */
      SparseFloatTensor() {}
      /** Construct a sparse tensor from a vector of 2D sparse mats */
      /** Stacking the 2D sparse matrices along the batch dimension, resulting in a 3D tensor */
      SparseFloatTensor(const std::vector<SpMat> & sparse2Ds, bool squeeze_batch = true);

      /** Construct a sparse tensor from shapes, values, and dimensions */
      SparseFloatTensor(Array<DimensionType> & shape,
          Array<DataType> & values,
          std::vector<DimData> & dims) :
              shape_(std::move(shape)),
              values_(std::move(values)), dims_(std::move(dims)) {}

      /** Copy is not supported */
      SparseFloatTensor(const SparseFloatTensor& other) = delete;
      /** Copy assignment is not supported */
      SparseFloatTensor& operator=(const SparseFloatTensor& other) = delete;

      /** Move constructor */
      SparseFloatTensor(SparseFloatTensor&& other) noexcept :
            shape_(std::move(other.shape_)), values_(std::move(other.values_)),
            dims_(std::move(other.dims_)) {}

      /** Move assignment */
      SparseFloatTensor& operator=(SparseFloatTensor&& other) noexcept
      {
        shape_ = std::move(other.shape_);
        values_ = std::move(other.values_);
        dims_ = std::move(other.dims_);
        return *this;
      }

      /** get the shape_ */
      Array<DimensionType> & shape() { return shape_; }
      /** get a const reference for shape_ */
      const Array<DimensionType> & shape() const { return shape_; }
      /** get the values */
      Array<DataType> & values() { return values_; }
      /** get a const reference for values_ */
      const Array<DataType> & values() const { return values_; }
      /** get the dimensions */
      std::vector<DimData> & dims() { return dims_; }
      /** get a const reference for dims_ */
      const std::vector<DimData> & dims() const { return dims_; }

      /** Construct a vector of MemWrapper on SpMatMap */
      std::vector<MemWrapper<SpMatMap>> toSparse2Ds();

      #ifdef DEBUG
      void checkShapeAndDim() {
        // For 1D, it should be represented as 2D while have the first
        // dimension as size 1.
        Require(shape_.size() >= 2 && (shape_.size() == dims_.size() + 1), "The shape size should be bigger or equal to 2 and should be consistent with dims size + 1");
        Require(dims_[0].outer().size() == shape_[0] + 1, "The first dimension should be dense.");
        for (size_t i=0; i<dims_.size(); i++) dims_[i].checkDataCorrectness();
        Require(values_.size() == dims_[shape_.size() - 2].inner().size(), "The last dimension should include all the non-zeros.");
      }
      #endif // DEBUG
  };

} // namespace ops

#endif // OPS_SPARSEFLOATTENSOR_H_
