/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.StridedUtils.layoutFromShapeStrides
import org.diffkt.StridedUtils.Layout
import org.diffkt.StridedUtils.contigStrides
import org.diffkt.StridedUtils.strided
import shapeTyping.annotations.SType

/**
 * An int tensor, which stores its underlying data in an int array and tracks
 * the strides needed to access data for each dimension.
 */
@SType("S: Shape")
class IntTensor internal constructor(
        val shape: @SType("S") Shape,
        internal val offset: Int,
        internal val strides: IntArray,
        internal val data: IntArray,
        internal val layout: Layout = layoutFromShapeStrides(data.size, shape, strides),
) {
    val rank get() = shape.rank
    val size get() = shape.product

    operator fun get(index: Int): IntTensor = view(index, axis = 0)

    fun at(pos: Int): Int {
        return when (layout) {
            Layout.NATURAL -> data[pos + offset]
            Layout.SINGLETON -> data[offset]
            Layout.REPEATING -> data[(pos + offset) % data.size]
            Layout.CUSTOM -> data[strided(pos)]
        }
    }

    val dataIterator: Iterable<Int> get() = object : Iterable<Int> {
        override fun iterator(): Iterator<Int> = object : Iterator<Int> {
            var curr = 0
            override fun hasNext(): Boolean = curr < size
            override fun next(): Int =
                    if (hasNext()) at(curr++) else throw NoSuchElementException("$this has no next element")
        }
    }

    /**
     * Return an equivalent tensor whose representation is an [IntTensor] with natural layout
     * and zero offset (that is, with a contiguous data representation).
     */
    open fun normalize(): IntTensor {
        return IntTensor(this.shape) { i -> this.at(i) }
    }

    fun view(index: Int, axis: Int): IntTensor {
        require(axis >= 0 && axis < shape.rank)
        require(index >= 0 && index < shape[axis])
        val newShape = shape.remove(axis)
        val newStrides = strides.remove(axis)
        return IntTensor(newShape, offset + index * strides[axis], newStrides, data)
    }

    private fun strided(contig: Int): Int {
        return strided(contig, shape, strides) + offset
    }

    fun flatten(startDim: Int = 0, endDim: Int = rank - 1): IntTensor {
        if (startDim >= endDim) return this
        val flattenedDim = (startDim..endDim).fold(1, { acc, nextDim -> acc * shape[nextDim] })
        val newDims = shape.take(startDim) + flattenedDim + shape.drop(endDim + 1)
        return reshape(newDims)
    }

    fun reshape(newShape: Shape): IntTensor {
        // Check that new shape is valid.
        val oldTensor = this
        val oldShape = oldTensor.shape
        if (newShape == oldShape) return oldTensor
        require(oldShape.product() == newShape.product())

        return when (layout) {
            Layout.SINGLETON ->
                IntTensor(shape, offset, strides = StridedUtils.singletonStrides(shape.rank), data, layout)
            Layout.NATURAL ->
                IntTensor(shape, offset, strides = StridedUtils.contigStrides(shape), data, layout)
            else -> normalize().reshape(shape)
        }
    }

    companion object {
        operator fun invoke(shape: Shape, values: IntArray): IntTensor {
            assert(shape.product == values.size)
            return IntTensor(shape, offset = 0, contigStrides(shape), values, Layout.NATURAL)
        }

        operator fun invoke(shape: Shape, generator: (Int) -> Int): IntTensor {
            if (shape.isScalar) return intScalarOf(generator(0))
            return IntTensor(shape, IntArray(shape.product(), generator))
        }
    }
}

fun intTensorOf(vararg data: Int) = IntTensor(Shape(data.size), data)

fun intScalarOf(value: Int) = IntTensor(Shape(), intArrayOf(value))
