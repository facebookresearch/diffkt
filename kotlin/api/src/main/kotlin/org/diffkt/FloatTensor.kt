/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.gpu.GpuFloatTensor
import kotlin.random.Random
import shapeTyping.annotations.SType

@SType("S: Shape")
abstract class FloatTensor : @SType("S") DTensor {
    override val derivativeID: DerivativeID get() = NoDerivativeID
    override val primal: @SType("S") DTensor get() = this
    abstract override val shape: @SType("S") Shape
    abstract fun at(pos: Int): Float

    fun gpu(): @SType("S") GpuFloatTensor {
        return when (this) {
            is GpuFloatTensor -> this
            is StridedFloatTensor -> GpuFloatTensor(shape, normalize().data)
            is FloatScalar -> GpuFloatTensor(Shape(), floatArrayOf(this.value))
            else -> throw NotImplementedError("GPU tensor conversion currently unavailable for ${this::class}")
        }
    }

    open fun cpu(): @SType("S") FloatTensor {
        return when (this) {
            is GpuFloatTensor -> this.cpu()
            else -> this
        }
    }

    fun to(device: Device): @SType("S") FloatTensor {
        return when (device) {
            Device.CPU -> cpu()
            Device.GPU -> gpu()
        }
    }

    val device: Device get() = if (this is GpuFloatTensor) Device.GPU else Device.CPU

    open fun map(f: (Float)->Float): @SType("S") FloatTensor = FloatTensor(shape) { pos -> f(at(pos)) }
    open fun mapIndexed(f: (Int, Float) -> Float): @SType("S") FloatTensor = FloatTensor(shape) { pos -> f(pos, at(pos)) }
    open fun zip(right: @SType("S") FloatTensor, f: (Float, Float)->Float): @SType("S") FloatTensor {
        assert(this.shape == right.shape)
        return FloatTensor(shape) { pos -> f(at(pos), right.at(pos)) }
    }

    /**
     * map for when the function is not pure.
     */
    open fun impureMap(f: (Float)->Float): @SType("S") FloatTensor = FloatTensor(shape) { pos -> f(at(pos)) }

    /**
     * zip for when the function is not pure.
     */
    open fun impureZip(right: @SType("S") FloatTensor, f: (Float, Float)->Float): @SType("S") FloatTensor {
        assert(this.shape == right.shape)
        return FloatTensor(shape) { pos -> f(at(pos), right.at(pos)) }
    }

    open fun zip2(
        second: @SType("S") FloatTensor,
        third: @SType("S") FloatTensor,
        f: (Float, Float, Float)->Float
    ): @SType("S") FloatTensor {
        assert(this.shape == second.shape && this.shape == third.shape)
        return FloatTensor(shape) { pos -> f(at(pos), second.at(pos), third.at(pos)) }
    }

    /**
     * zip2 for when the function is not pure.
     */
    open fun impureZip2(
        second: @SType("S") FloatTensor,
        third: @SType("S") FloatTensor,
        f: (Float, Float, Float)->Float
    ): @SType("S") FloatTensor {
        assert(this.shape == second.shape && this.shape == third.shape)
        return FloatTensor(shape) { pos -> f(at(pos), second.at(pos), third.at(pos)) }
    }

    val allAxes: IntArray
        get() = IntArray(rank) { it }

    fun reduce(f: (Float, Float) -> Float, axes: IntArray = allAxes, keepDims: Boolean = false): FloatTensor =
        Combinators.reduce(this, f, axes, keepDims)

    open fun all(p: (Float) -> Boolean): Boolean {
        for (i in 0 until size) {
            if (!p(this.at(i))) return false
        }
        return true
    }

    private fun toString(builder: StringBuilder) {
        if (this is FloatScalar) {
            builder.append(this.value.toString())
            return
        }

        builder.append('[')
        for (i in 0 until this.shape[0]) {
            if (i > 0) builder.append(", ")
            (this[i] as FloatTensor).toString(builder)
        }
        builder.append(']')
    }
    override fun toString(): String {
        val b = StringBuilder()
        this.toString(b)
        return b.toString()
    }

    override fun toCodeString(): String {
        if (this is FloatScalar)
            return "FloatScalar(${this.value}f)"

        val b = StringBuilder()
        b.append("tensorOf(")
        var i: Int = 0
        for (ix in indices) {
            when {
                i == 0 -> if (size > 9) b.append("\n");
                (i % 10) == 0 -> b.append(",\n");
                else -> b.append(", ")
            }
            b.append(this.at(i).toString() + "f")
            i++
        }
        if (shape.rank == 1)
            b.append(")")
        else
            b.append(").reshape($shape)")
        return b.toString()
    }

    /**
     * Return an equivalent tensor whose representation is a [StridedFloatTensor] with natural layout
     * and zero offset (that is, with a contiguous data representation).
     */
    open fun normalize(): @SType("S") StridedFloatTensor {
        return FloatTensor(this.shape) { i -> this.at(i) } as StridedFloatTensor
    }

    fun asStrided(): @SType("S") StridedFloatTensor {
        return if (this is StridedFloatTensor) this else normalize()
    }

    fun asList(): List<Float> {
        return List(size) {this.at(it)}
    }

    /**
     * Computes the tensor index corresponding to the position in normal form.
     */
    fun posToIndex(contig: Int): IntArray {
        val rank = shape.rank
        val index = IntArray(rank)
        var ctg = contig
        var i = rank - 1
        while (i >= 0) {
            val wid = shape[i]
            val off = ctg % wid
            ctg /= wid
            index[i] = off
            i -= 1
        }
        return index
    }

    internal open fun getAt(ix: IntArray): Float = at(indexToPos(ix))

    internal fun getAtIndex(vararg ix: Int): Float = getAt(ix)

    /**
     * Computes the position in normal form for a given tensor index.
     */
    internal fun indexToPos(index: IntArray): Int {
        require(index.size == rank)
        require(index.indices.all { index[it] >= 0 && index[it] < shape[it] })
        var stride = 1
        var pos = 0
        for (i in index.indices.reversed()) {
            pos += stride * index[i]
            stride *= shape[i]
        }
        return pos
    }

    // --- Equality ---
    override fun equals(other: Any?): Boolean =
        this === other ||
            other is FloatTensor &&
            other.shape == shape &&
            (0 until size).all { at(it) == other.at(it) }

    override fun hashCode(): Int {
        return (0 until size).fold("FloatTensor".hashCode()) { h: Int, x: Int ->
            val f = at(x)
            h.shl(5) + h + f.hashCode()
        }
    }

    companion object {
        @JvmName("vararg ctor")
        @SType("S: Shape")
        operator fun invoke(shape: @SType("S") Shape, vararg capturedValues: Float) : @SType("S") FloatTensor {
            return invoke(shape, capturedValues)
        }

        @SType("S: Shape")
        operator fun invoke(shape: @SType("S") Shape, capturedValues: FloatArray) : @SType("S") FloatTensor {
            if (shape.isScalar) return FloatScalar(capturedValues[0])
            return StridedFloatTensor(shape, capturedValues)
        }

        @SType("S: Shape")
        operator fun invoke(shape: @SType("S") Shape, generator: (Int) -> Float): @SType("S") FloatTensor {
            if (shape.isScalar) return FloatScalar(generator(0))
            val i = shape.product
            return StridedFloatTensor(shape, FloatArray(i, generator))
        }

        @SType("S: Shape")
        fun random(random: Random, shape: @SType("S") Shape, min: Float = 0f, max: Float = 1f): @SType("S") FloatTensor {
            if (shape.isScalar) return FloatScalar(random.nextFloat() * (max - min) + min)
            val values = FloatArray(shape.product()) { random.nextFloat() * (max - min) + min }
            return FloatTensor(shape, values)
        }

        @SType("S: Shape")
        fun zeros(shape: @SType("S") Shape): @SType("S") FloatTensor {
            return if (shape.isScalar)
                FloatScalar(0F)
            else
                StridedFloatTensor(
                        shape,
                        offset = 0,
                        StridedUtils.singletonStrides(shape.rank),
                        floatArrayOf(0f),
                        StridedUtils.Layout.SINGLETON)
        }

        // TODO: remove this operation and move it to an Operations implementation.
        @SType("S: Shape")
        fun ones(shape: @SType("S") Shape): @SType("S") FloatTensor {
            return if (shape.isScalar)
                FloatScalar(1f)
            else
                StridedFloatTensor(
                        shape,
                        offset = 0,
                        StridedUtils.singletonStrides(shape.rank),
                        floatArrayOf(1f),

                        StridedUtils.Layout.SINGLETON)
        }

        @SType("S: Shape")
        fun const(value: Float, shape: @SType("S") Shape): @SType("S") FloatTensor {
            return if (shape.isScalar)
                FloatScalar(value)
            else
                StridedFloatTensor.singleton(shape, value)
        }

    }
}
