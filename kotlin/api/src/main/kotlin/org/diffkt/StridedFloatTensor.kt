/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.StridedUtils.layoutFromShapeStrides
import org.diffkt.StridedUtils.Layout
import org.diffkt.StridedUtils.strided
import org.diffkt.StridedUtils.stridesAt
import org.diffkt.StridedUtils.contigStrides
import org.diffkt.StridedUtils.singletonStrides
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

/**
 * A float tensor, which stores its underlying data in a float array and tracks
 * the strides needed to access data for each dimension.
 */
@SType("S: Shape")
@AllowUnreduced
class StridedFloatTensor internal constructor(
        override val shape: @SType("S") Shape,
        internal val offset: Int,
        internal val strides: IntArray,
        internal val data: FloatArray,
        internal val layout: Layout = layoutFromShapeStrides(data.size, shape, strides),
) : FloatTensor() {
    init {
        assert(!shape.isScalar)
        FloatArrayPool.incrRefCount(data)
    }
    override val derivativeID: DerivativeID get() = NoDerivativeID
    override val primal: DTensor get() = this

    protected fun finalize() {
        // Hang onto the underlying data array since creating new FloatArrays is expensive
        FloatArrayPool.put(data)
    }

    // --- Data Access/Indexing ---
    override fun at(pos: Int): Float {
        return when (layout) {
            Layout.NATURAL -> data[pos + offset]
            Layout.SINGLETON -> data[offset]
            Layout.REPEATING -> data[(pos + offset) % data.size]
            Layout.CUSTOM -> data[strided(pos)]
        }
    }

    override fun normalize(): @SType("S") StridedFloatTensor {
        return when (layout) {
            Layout.SINGLETON -> data[offset].let { d -> FloatTensor(this.shape) { d } } as StridedFloatTensor
            Layout.REPEATING -> super.normalize()
            Layout.NATURAL -> if (offset == 0) this else FloatTensor(shape, data.sliceArray(offset until offset + size)) as StridedFloatTensor
            else -> this.map { it }
        }
    }

    override fun all(p: (Float) -> Boolean): Boolean {
        when (layout) {
            Layout.SINGLETON -> return p(data[offset])
            Layout.REPEATING -> return (0 until kotlin.math.min(data.size,this.size)).all { p(data[(it + offset) % data.size]) }
            Layout.NATURAL -> return (0 until this.size).all { p(data[it + offset]) }
            Layout.CUSTOM -> {
                fun allAxis(offset: Int, axis: Int): Boolean {
                    val stride = strides[axis]
                    var o = offset
                    for (index in 0 until shape[axis]) {
                        val r = if (axis == shape.rank - 1) {
                            p(this.data[o])
                        } else {
                            allAxis(o, axis + 1)
                        }
                        if (!r) return false
                        o += stride
                    }
                    return true
                }
                return allAxis(this.offset, axis = 0)
            }
        }
    }

    override fun map(f: (Float)->Float): @SType("S") StridedFloatTensor {
        return when (layout) {
            Layout.SINGLETON -> StridedFloatTensor(shape, offset = 0, strides = IntArray(shape.rank), data = FloatArray(1) { f(data[offset]) }, layout = Layout.SINGLETON)
            Layout.REPEATING -> StridedFloatTensor(shape, offset = this.offset, strides = this.strides, data = FloatArray(data.size) { f(data[it] ) }, layout = Layout.REPEATING)
            Layout.NATURAL -> StridedFloatTensor(shape, offset = 0, strides = this.strides, data = FloatArray(this.size) { f(data[it + offset]) }, layout = Layout.NATURAL)
            Layout.CUSTOM -> {
                val newData = FloatArray(this.size)
                var next: Int = 0
                fun copyAxis(offset: Int, axis: Int) {
                    val stride = strides[axis]
                    var o = offset
                    for (index in 0 until shape[axis]) {
                        if (axis == shape.rank - 1) {
                            newData[next++] = f(this.data[o])
                        } else {
                            copyAxis(o, axis + 1)
                        }
                        o += stride
                    }
                }
                copyAxis(this.offset, axis = 0)
                assert(next == newData.size)
                StridedFloatTensor(shape, newData)
            }
        }
    }

    override fun impureMap(f: (Float)->Float): @SType("S") StridedFloatTensor {
        return when (layout) {
            Layout.NATURAL -> StridedFloatTensor(shape, offset = 0, strides = this.strides, data = FloatArray(this.size) { f(data[it + offset]) }, layout = Layout.NATURAL)
            Layout.CUSTOM -> {
                val newData = FloatArray(this.size)
                var next: Int = 0
                fun copyAxis(offset: Int, axis: Int) {
                    val stride = strides[axis]
                    var o = offset
                    for (index in 0 until shape[axis]) {
                        if (axis == shape.rank - 1) {
                            newData[next++] = f(this.data[o])
                        } else {
                            copyAxis(o, axis + 1)
                        }
                        o += stride
                    }
                }
                copyAxis(this.offset, axis = 0)
                assert(next == newData.size)
                StridedFloatTensor(shape, newData)
            }
            else -> super.impureMap(f) as StridedFloatTensor
        }
    }

    override fun zip(right: @SType("S") FloatTensor, f: (Float, Float)->Float): @SType("S") FloatTensor {
        val left = this
        assert(left.shape == right.shape)
        if (right !is StridedFloatTensor || right.layout == Layout.REPEATING)
            return super.zip(right, f)
        if (right.layout == Layout.SINGLETON)
            return right.data[right.offset].let { rightData -> left.map { f(it, rightData) } }
        return when (left.layout) {
            Layout.SINGLETON -> left.data[left.offset].let { thisData -> right.map { f(thisData, it) } }
            Layout.REPEATING -> super.zip(right, f)
            else -> { // Layout.CUSTOM or Layout.NATURAL
                val newData = FloatArray(left.size)
                var nextDataIndex: Int = 0
                fun copyAxis(leftOffset: Int, rightOffset: Int, axis: Int) {
                    val leftStride = left.strides[axis]
                    val rightStride = right.strides[axis]
                    var _leftOffset = leftOffset
                    var _rightOffset = rightOffset
                    for (index in 0 until left.shape[axis]) {
                        if (axis == left.shape.rank - 1) {
                            newData[nextDataIndex++] = f(left.data[_leftOffset], right.data[_rightOffset])
                        } else {
                            copyAxis(_leftOffset, _rightOffset, axis + 1)
                        }
                        _leftOffset += leftStride
                        _rightOffset += rightStride
                    }
                }
                copyAxis(left.offset, right.offset, axis = 0)
                assert(nextDataIndex == newData.size)
                StridedFloatTensor(shape, newData)
            }
        }
    }

    override fun impureZip(right: @SType("S") FloatTensor, f: (Float, Float)->Float): @SType("S") FloatTensor {
        val left = this
        assert(left.shape == right.shape)
        if (right !is StridedFloatTensor || right.layout == Layout.REPEATING)
            return super.impureZip(right, f)
        if (right.layout == Layout.SINGLETON)
            return right.data[right.offset].let { rightData -> left.impureMap { f(it, rightData) } }
        return when (left.layout) {
            Layout.SINGLETON -> left.data[left.offset].let { thisData -> right.impureMap { f(thisData, it) } }
            Layout.REPEATING -> super.impureZip(right, f)
            else -> { // Layout.CUSTOM or Layout.NATURAL
                val newData = FloatArray(left.size)
                var nextDataIndex: Int = 0
                fun copyAxis(leftOffset: Int, rightOffset: Int, axis: Int) {
                    val leftStride = left.strides[axis]
                    val rightStride = right.strides[axis]
                    var _leftOffset = leftOffset
                    var _rightOffset = rightOffset
                    for (index in 0 until left.shape[axis]) {
                        if (axis == left.shape.rank - 1) {
                            newData[nextDataIndex++] = f(left.data[_leftOffset], right.data[_rightOffset])
                        } else {
                            copyAxis(_leftOffset, _rightOffset, axis + 1)
                        }
                        _leftOffset += leftStride
                        _rightOffset += rightStride
                    }
                }
                copyAxis(left.offset, right.offset, axis = 0)
                assert(nextDataIndex == newData.size)
                StridedFloatTensor(shape, newData)
            }
        }
    }

    override fun zip2(
        second: @SType("S") FloatTensor,
        third: @SType("S") FloatTensor,
        f: (Float, Float, Float)->Float
    ): @SType("S") FloatTensor {
        val first = this
        assert(first.shape == second.shape && second.shape == third.shape)
        if (second !is StridedFloatTensor || second.layout == Layout.REPEATING ||
            third !is StridedFloatTensor || third.layout == Layout.REPEATING)
            return super.zip2(second, third, f)
        if (second.layout == Layout.SINGLETON)
            return second.data[second.offset].let { b -> first.zip(third) { a, c -> f(a, b, c) } }
        if (third.layout == Layout.SINGLETON)
            return third.data[third.offset].let { c -> first.zip(second) { a, b -> f(a, b, c) } }

        return when (first.layout) {
            Layout.SINGLETON -> first.data[first.offset].let { a -> second.zip(third) { b, c -> f(a, b, c) } }
            Layout.REPEATING -> super.zip2(second, third, f)
            else -> { // Layout.CUSTOM or Layout.NATURAL
                val newData = FloatArray(first.size)
                var nextDataIndex: Int = 0
                fun copyAxis(firstOffset: Int, secondOffset: Int, thirdOffset: Int, axis: Int) {
                    val firstStride = first.strides[axis]
                    val secondStride = second.strides[axis]
                    val thirdStride = third.strides[axis]
                    var _firstOffset = firstOffset
                    var _secondOffset = secondOffset
                    var _thirdOffset = thirdOffset
                    for (index in 0 until first.shape[axis]) {
                        if (axis == first.shape.rank - 1) {
                            newData[nextDataIndex++] = f(first.data[_firstOffset], second.data[_secondOffset], third.data[_thirdOffset])
                        } else {
                            copyAxis(_firstOffset, _secondOffset, _thirdOffset, axis + 1)
                        }
                        _firstOffset += firstStride
                        _secondOffset += secondStride
                        _thirdOffset += thirdStride
                    }
                }
                copyAxis(first.offset, second.offset, third.offset, axis = 0)
                assert(nextDataIndex == newData.size)
                StridedFloatTensor(shape, newData)
            }
        }
    }

    override fun impureZip2(
        second: @SType("S") FloatTensor,
        third: @SType("S") FloatTensor,
        f: (Float, Float, Float)->Float
    ): @SType("S") FloatTensor {
        val first = this
        assert(first.shape == second.shape && second.shape == third.shape)
        if (second !is StridedFloatTensor || second.layout == Layout.REPEATING ||
            third !is StridedFloatTensor || third.layout == Layout.REPEATING)
            return super.impureZip2(second, third, f)
        if (second.layout == Layout.SINGLETON)
            return second.data[second.offset].let { b -> first.impureZip(third) { a, c -> f(a, b, c) } }
        if (third.layout == Layout.SINGLETON)
            return third.data[third.offset].let { c -> first.impureZip(second) { a, b -> f(a, b, c) } }

        return when (first.layout) {
            Layout.SINGLETON -> first.data[first.offset].let { a -> second.impureZip(third) { b, c -> f(a, b, c) } }
            Layout.REPEATING -> super.impureZip2(second, third, f)
            else -> { // Layout.CUSTOM or Layout.NATURAL
                val newData = FloatArray(first.size)
                var nextDataIndex: Int = 0
                fun copyAxis(firstOffset: Int, secondOffset: Int, thirdOffset: Int, axis: Int) {
                    val firstStride = first.strides[axis]
                    val secondStride = second.strides[axis]
                    val thirdStride = third.strides[axis]
                    var _firstOffset = firstOffset
                    var _secondOffset = secondOffset
                    var _thirdOffset = thirdOffset
                    for (index in 0 until first.shape[axis]) {
                        if (axis == first.shape.rank - 1) {
                            newData[nextDataIndex++] = f(first.data[_firstOffset], second.data[_secondOffset], third.data[_thirdOffset])
                        } else {
                            copyAxis(_firstOffset, _secondOffset, _thirdOffset, axis + 1)
                        }
                        _firstOffset += firstStride
                        _secondOffset += secondStride
                        _thirdOffset += thirdStride
                    }
                }
                copyAxis(first.offset, second.offset, third.offset, axis = 0)
                assert(nextDataIndex == newData.size)
                StridedFloatTensor(shape, newData)
            }
        }
    }

    /**
     * Helper function for validating the number of indices to index into this tensor.
     *
     * Throws an IllegalArugmentException if the number of indices does not match this
     * tensor's rank.
     */
    private fun checkIndicesSize(indices: IntArray) {
        require(shape.rank == indices.size) {
            "Must index Tensor of rank ${shape.rank} with ${shape.rank} indices; got ${indices.size}" }
    }

    override internal fun getAt(ix: IntArray): Float =
        data[ix.foldIndexed(offset, { index, acc, i -> acc + i * stridesAt(strides, index) })]

    override val operations: Operations
        get() = StridedFloatTensorOperations

    private fun strided(contig: Int): Int {
        return strided(contig, shape, strides) + offset
    }

    /**
     * Helper function for validating the number and value of indices to index into
     * this tensor.
     *
     * Throws if checkIndicesSize throws or if any index is invalid for this Tensor's
     * shape.
     */
    private fun checkIndices(indices: IntArray) {
        checkIndicesSize(indices)
        for (i in indices.indices) {
            if (indices[i] >= shape[i] || indices[i] < 0) {
                throw IndexOutOfBoundsException("attempted to index array of shape $shape with ${indices.toList()}")
            }
        }
    }

    companion object {
        @SType("S: Shape")
        internal fun singleton(shape: @SType("S") Shape, value: Float): @SType("S") StridedFloatTensor {
            require(shape.rank != 0)
            return StridedFloatTensor(shape, offset = 0, singletonStrides(shape.rank), floatArrayOf(value), Layout.SINGLETON)
        }

        @SType("S: Shape")
        fun contigZeros(shape: @SType("S") Shape): @SType("S") StridedFloatTensor {
            return StridedFloatTensor(shape, FloatArrayPool.get(shape, clear = true))
        }

        /**
         * Creates a StridedFloatTensor with a contiguous/natural layout populated by populateFn.
         *
         * The FloatArray to populate may not be zeroed.
         */
        @SType("S: Shape")
        fun contiguous(shape: @SType("S") Shape, populateFn: (FloatArray) -> Unit): @SType("S") StridedFloatTensor {
            val data = FloatArrayPool.get(shape, clear = false)
            populateFn(data)
            return StridedFloatTensor(shape, data)
        }

        @SType("S: Shape")
        operator fun invoke(shape: @SType("S") Shape, values: FloatArray): @SType("S") StridedFloatTensor {
            assert(shape.product == values.size)
            return StridedFloatTensor(shape, offset = 0, contigStrides(shape), values, Layout.NATURAL)
        }

        @SType("S: Shape")
        fun identityGradient(halfShape: @SType("S") Shape): @SType("concat(S, S)") FloatTensor {
            if (halfShape.isScalar)
                return FloatScalar.ONE
            val size = halfShape.product()
            val sizep1 = size + 1
            return FloatTensor(halfShape + halfShape) { i: Int -> if (i % sizep1 == 0) 1F else 0F }
        }
    }
}
