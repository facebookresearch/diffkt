/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import java.util.concurrent.atomic.AtomicInteger

internal object FloatArrayPool {
    // Config
    private val maxPerSize = 50
    private val totalMax = 1000
    // To disable pooling, change minSize to Int.MAX_VALUE
    private val minSize = 128 // Appears to be fastest for Resnet when compared to 64 annd 256

    // Data
    private var totalElems = AtomicInteger()
    private val pool: MutableMap<Int, MutableSet<FloatArray>> = HashMap()
    val refCounts: MutableMap<FloatArray, Int> = HashMap()

    // private lock for synchronized block
    private val lock = Any()

    private fun <T> MutableSet<T>.popOrNull(): T? = firstOrNull()?.also { remove(it) }

    private fun dumpHistogram() {
        synchronized(lock) {
            println("--- BEGIN HISTOGRAM ---")
            pool.forEach { (k, v) -> println("${k}\t${v.size}") }
            println("--- END HISTOGRAM ---")
        }
    }

    fun get(shape: Shape, clear: Boolean): FloatArray = get(shape.product, clear)

    fun get(size: Int, clear: Boolean): FloatArray {
        if (size < minSize) return FloatArray(size)
        synchronized(lock) {
            val removed = pool[size]?.popOrNull()
            if (removed != null) {
                totalElems.decrementAndGet()
            }
            val arr = removed ?: FloatArray(size)
            // Initialize the refcount entry for this array, making it
            // so we track refcounts for this array, which we know is
            // larger than the min size
            if (!refCounts.containsKey(arr)) refCounts[arr] = 0
            // If we got an array from the pool and the caller requires a cleared array, clear it.
            if (clear && removed != null) {
                for (i in 0 until removed.size) removed[i] = 0F
            }
            return arr
        }
    }

    fun incrRefCount(arr: FloatArray) {
        synchronized(lock) {
            if (refCounts.containsKey(arr)) {
                refCounts[arr] = refCounts[arr]!! + 1
            }
        }
    }

    fun put(arr: FloatArray) {
        val size = arr.size
        if (size < minSize) return
        synchronized(lock) {
            if (!refCounts.containsKey(arr)) return
            val oldRefCount = refCounts[arr]!!
            if (oldRefCount == 1) {
                refCounts.remove(arr)
            } else {
                refCounts[arr] = oldRefCount - 1
            }

            if (totalElems.get() >= totalMax) return
            if (!pool.containsKey(size)) pool[size] = mutableSetOf()
            if (pool[size]!!.size >= maxPerSize) return

            if (oldRefCount == 1) {
                totalElems.incrementAndGet()
                assert(arr !in pool[size]!!)
                pool[size]!! += arr
            }
        }
    }
}
