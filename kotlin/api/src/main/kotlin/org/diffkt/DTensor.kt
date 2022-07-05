/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import shapeTyping.annotations.SType
import org.diffkt.adOptimize.DTensorRoot

/**
 * Interface for a differentiable tensor.
 *
 * [DTensor] is an interface that is implemented as either
 * - a [FloatTensor], which is a wrapper around an array of floats, or
 * - a [ForwardTensor] for forward differentiation, or
 * - a [ReverseTensor] for reverse mode differentiation.
 */
@DTensor
@SType("S: Shape")
interface DTensor: Differentiable<DTensor> {

    /**
     * Each derivative is assigned a unique DerivativeID
     **/
    val derivativeID: DerivativeID

    /**
     * shape indicates the number of dimension of a tensor and the length of each dimension.
     * If the shape of the tensor is 3x4x5 then the value of shape is Shape(3,4,5).
     * @sample org.diffkt.samples.DTensorSamples.showShape
     * */
    val shape: @SType("S") Shape get() = primal.shape

    /**
     * primal points to the actual tensor
     * @sample org.diffkt.samples.DTensorSamples.showPrimal
     * */
    val primal: @SType("S") DTensor

    /**
     * The number of dimensions in the tensor's shape.
     * rank 0 - Scalar
     *  rank 1 - 1D array or 1D tensor
     *  rank 2 - 2D matrix or 2D tensor
     *  rank 3 - 3D tensor
     *  ...
     *  rank N - ND tensor
     *  @sample org.diffkt.samples.DTensorSamples.showRank
     *  */
    val rank: Int get() = shape.rank

    /**
     * The total number of elements of this tensor.
     * @sample org.diffkt.samples.DTensorSamples.showSize
     * */
    val size: Int get() = shape.product()

    /**
     * True if the tensor is a scalar.
     * @sample org.diffkt.samples.DTensorSamples.showIsScalar
     * */
    val isScalar: Boolean get() = shape.isScalar

    /**
     * The operations available on a tensor.
     * */
    val operations: Operations

    /**
     * Wrapper around the tensor
     */
    override fun wrap(wrapper: Wrapper): @SType("S") DTensor {
        return wrapper.wrapDTensor(this)
    }

    fun toCodeString(): String = toString()

    operator fun get(index: Int): DTensor = view(index, axis = 0)
    
    operator fun get(vararg indices: Int): DTensor = view(indices)

    /**
     * An iterator over the indices for the tensor
     * */
    val indices: Iterator<IntArray> get() = object : Iterator<IntArray> {
        // Start at intArrayOf(0, 0, ..., -1)
        private val curr = IntArray(rank) { if (it == rank - 1) -1 else 0 }
        private var scalarIsDone = false
        private val lastIxs = shape.dims.map { it - 1 }.toIntArray()

        override fun hasNext(): Boolean {
            return if (isScalar) !scalarIsDone
            // Stop when we are at listOf(shape[0]-1, shape[1]-1, ...)
            else !curr.contentEquals(lastIxs)
        }

        override fun next(): IntArray {
            // Special case handling for scalar tensors; we spit out listOf() once.
            if (isScalar) {
                scalarIsDone = true
                return intArrayOf()
            }
            // Increment right-most index; if we are too large, reset to zero
            // and increment the next index.
            var ix = rank - 1
            while (ix >= 0) {
                curr[ix]++
                if (curr[ix] < shape[ix])
                    break
                else
                    curr[ix--] = 0
            }
            return curr
        }
    }
}
