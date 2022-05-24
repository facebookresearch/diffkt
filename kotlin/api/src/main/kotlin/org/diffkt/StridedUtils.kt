/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

internal object StridedUtils {
    /** Data access layouts for Tensors
     *
     *  NATURAL   = Elements "fill in" along the axes from innermost (rightmost) axes out. Each element corresponds
     *              to one address of the Tensor.
     *  SINGLETON = All elements are identical, so we use a single-item array for backing data
     *  REPEATING = The same set of elements is used, in the same order, so we only need store one cycle of the elements.
     *  CUSTOM    = Element access order determined by the specified strides.
     */
    internal enum class Layout {
        NATURAL,
        SINGLETON,
        REPEATING,
        CUSTOM,
    }

    /** use this for contiguous vector stride */
    private val vectorStride: IntArray = intArrayOf(1)

    /** use these for singleton tensors */
    internal fun singletonStrides(rank: Int) = IntArray(rank) { 0 }

    /**
     * Strides for a given shape over contiguous storage
     *
     * Example:
     *   >>> contigStrides(Shape(2, 3, 4))
     *   List(12, 4, 1)
     */
    fun contigStrides(shape: Shape): IntArray {
        return when (val rank = shape.rank) {
            1 -> vectorStride
            else -> {
                val str = IntArray(rank) { 0 }
                var w = 1
                for (i in rank - 1 downTo 0) {
                    str[i] = w
                    w *= shape[i]
                }
                str
            }
        }
    }

    internal fun strided(contig: Int, shape: Shape, strides: IntArray): Int {
        val rank = shape.rank
        var ctg = contig
        var str = 0
        var i = rank - 1
        while (i >= 0) {
            val wid = shape[i]
            val off = ctg % wid
            ctg /= wid
            str += off * stridesAt(strides, i)
            i -= 1
        }
        return str
    }

    internal fun stridesAt(strides: IntArray, i: Int): Int =
            if (strides.size < i + 1) 0 else strides[i]

    /** size needed to hold data given repetition from broadcasting */
    fun dataSize(shape: Shape, strides: IntArray): Int {
        var n = 1
        var i = 0
        while (i < shape.rank) {
            if (stridesAt(strides, i) > 0)
                n *= shape[i]
            i += 1
        }
        return n
    }

    /**
     * Given shape and strides, return the strided layout.
     *
     * Singleton: strides are zero.
     * Natural: latter dimensions are laid out closer in memory, with the last dimension
     *      being laid out contiguously in memory
     * Repeating: from right to left, the same stride values as in natural strides, then
     *      zero strides
     * Custom: None of the above.
     *
     * See test cases for examples.
     */
    internal fun layoutFromShapeStrides(
            dataSize: Int,
            shape: Shape,
            strides: IntArray): Layout {
        assert(shape.rank == strides.size)

        if (strides.all { it == 0 }) return Layout.SINGLETON

        val contiguousStrides = contigStrides(shape)
        if (strides.contentEquals(contiguousStrides)) return Layout.NATURAL

        val firstNonZero = strides.lastIndexOf(0) + 1
        if (firstNonZero != 0) {
            // Check the size of the proposed data, since that will be used as the modulus.
            val requiredDataSize = strides.indices.map { if (strides[it] == 0) 1 else shape[it] }.product()
            if (dataSize == requiredDataSize && strides.indices.all { strides[it] == if (it < firstNonZero) 0 else contiguousStrides[it] })
                return Layout.REPEATING
        }

        return Layout.CUSTOM
    }
}