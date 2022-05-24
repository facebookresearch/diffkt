/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.gpu

import org.diffkt.*
import org.diffkt.external.Gpu
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class GpuFloatScalar internal constructor(val handle: Long) : FloatTensor(), DScalar {
    constructor(value: Float) : this(Gpu.putFloatTensor(intArrayOf(), floatArrayOf(value)))

    override val shape: Shape = Shape()
    override val operations: Operations get() = GpuFloatScalarOperations
    override val primal: DScalar get() = this
    override val derivativeID: DerivativeID get() = NoDerivativeID
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

    companion object {
        // Reference counts for the underlying C++ tensors
        private val referenceCounts = mutableMapOf<Long, Int>()
        private val referenceCountsLock = ReentrantLock()
    }
}