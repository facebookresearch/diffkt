/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.external

import org.diffkt.Shape
import org.diffkt.SparseFloatTensor

object SparseOps : ExternalLib {
    private const val DYLIB_NAME = "libsparseops_jni"
    private var _isLoaded = false

    override val isLoaded get() = _isLoaded

    init {
        try {
            loadLib(DYLIB_NAME)
            _isLoaded = true
        } catch (e: Exception) { }
    }

    fun convertToCoo(shape: Shape, entries: List<Pair<IntArray, Float>>): SparseFloatTensor {
        return convertToCoo(
            shape.dims,
            entries.map { it.first[0] }.toIntArray(),
            entries.map { it.first[1] }.toIntArray(),
            entries.map { it.second }.toFloatArray()
        )
    }

    // --- External functions ---

    external fun add(left: SparseFloatTensor, right: SparseFloatTensor): SparseFloatTensor
    external fun times(left: SparseFloatTensor, right: SparseFloatTensor): SparseFloatTensor
    external fun sub(left: SparseFloatTensor, right: SparseFloatTensor): SparseFloatTensor
    external fun matdiv(left: SparseFloatTensor, right: SparseFloatTensor): SparseFloatTensor
    external fun matmul(left: SparseFloatTensor, right: SparseFloatTensor): SparseFloatTensor
    external fun transpose(tensor: SparseFloatTensor): SparseFloatTensor
    external fun convertToCoo(shape: IntArray, rows: IntArray, cols: IntArray, values: FloatArray): SparseFloatTensor
}
