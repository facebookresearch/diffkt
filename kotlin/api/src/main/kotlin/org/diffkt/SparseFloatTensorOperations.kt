/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.external.SparseOps

internal object SparseFloatTensorOperations: FloatTensorOperations() {
    override val name get() = "SparseFloatTensor"
    private fun wrap(value: DTensor): SparseFloatTensor {
        if (value is SparseFloatTensor) return value
        TODO("Cannot (automatically) convert to SparseFloatTensor")
    }

    override fun plus(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        return SparseOps.add(l, r)
    }

    override fun minus(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        return SparseOps.sub(l, r)
    }

    override fun times(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId == NoDerivativeID)
        if (left is SparseFloatTensor) {
            if (right is SparseFloatTensor) {
                return SparseOps.times(left, right)
            } else {
                require(right is FloatTensor)
                return left.zip(right) { l, r -> l * r }
            }
        } else {
            require(left is FloatTensor)
            require(right is SparseFloatTensor)
            return right.zip(left) { r, l -> l * r }
        }
    }

    override fun transpose(x: DTensor, axes: IntArray): DTensor {
        require(x is SparseFloatTensor)
        /** Currently only supports 2D or 3D with the last two dimensions transposed. */
        require (axes.size == 2 || axes.size == 3) { "Sparse Transpose only supports 2D/3D for now." }
        val newAxis = x.allAxes
        /** swap last two Axes */
        val s = x.allAxes.size
        newAxis[s - 1] = s - 2
        newAxis[s - 2] = s - 1
        require(axes.contentEquals(newAxis)) { "Sparse Transpose only supported on the last two axes." }
        return SparseOps.transpose(x)
    }
}
