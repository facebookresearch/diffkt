/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

internal object SparseRowFloatTensorOperations : FloatTensorOperations() {
    override val name: String get() = "SparseRowFloatTensor"
    private fun wrap(value: DTensor): FloatTensor {
        require(value is FloatTensor)
        return value
    }

    override fun times(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        val l = wrap(left)
        val r = wrap(right)
        return when {
            l is SparseRowFloatTensor ->
                l.qualifiedZip(r, true, true) {x, y -> x * y}
            r is SparseRowFloatTensor ->
                r.qualifiedZip(l, true, true) {x, y -> x * y}
            else ->
                super.times(left, right, derivativeId)
        }
    }
}