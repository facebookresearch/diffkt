/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.gpu

import org.diffkt.FloatScalar
import org.diffkt.FloatTensor
import org.diffkt.Operations
import org.diffkt.Shape
import kotlin.concurrent.withLock
import java.util.concurrent.locks.ReentrantLock
import org.diffkt.external.Gpu
import shapeTyping.annotations.SType

// TODO Make GPUFloatTensor not inherit Float Tensor https://github.com/facebookresearch/diffkt/issues/348
@SType("S: Shape")
class GpuFloatTensor internal constructor(val handle: Long) : @SType("S") FloatTensor() {
    constructor(shape: @SType("S") Shape, data: FloatArray) : this(Gpu.putFloatTensor(shape.dims, data))

    override val shape: @SType("S") Shape by lazy { Shape(Gpu.getShape(handle)) }

    override fun at(pos: Int): Float {
        throw NotImplementedError("Cannot get data from GPU, call .cpu() to transfer")
    }

    // --- Memory management ---

    init {
        // increment reference count
        referenceCounts[handle] = referenceCounts.getOrDefault(handle, 0) + 1
    }

    protected fun finalize() {
        val removed = referenceCountsLock.withLock {
            referenceCounts[handle] = referenceCounts[handle]!! - 1
            if (referenceCounts[handle] == 0) {
                referenceCounts.remove(handle)
                true
            } else {
                false
            }
        }
        if (removed) {
            Gpu.deleteHandle(handle)
        }
    }

    override fun cpu(): FloatTensor {
        return if (shape == Shape())
            FloatScalar(Gpu.getFloatData(handle)[0])
        else
            FloatTensor(shape, Gpu.getFloatData(handle))
    }

    override val operations: Operations
        get() = GpuFloatTensorOperations

    companion object {
        // Reference counts for the underlying C++ tensors
        private val referenceCounts = mutableMapOf<Long, Int>()
        private val referenceCountsLock = ReentrantLock()
    }
}
