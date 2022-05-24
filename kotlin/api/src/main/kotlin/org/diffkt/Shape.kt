/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

@SType("S: Shape")
class Shape internal constructor(val dims: IntArray) {
    companion object {
        private val emptyArray = intArrayOf()
        private val emptyShape : @SType("[]") Shape = Shape(emptyArray) as @SType("[]") Shape
        // since shapes are immutable, return an empty one from the cache
        operator fun invoke(): @SType("[]") Shape = emptyShape

        @SType("S: Shape")
        // Note: shapeTyping syntax for specifying a shape over varargs is incomplete.
        // This currently means that the vararg Ints in shapes are used as dims to make the shape S, but
        // it's likely to be reworked.
        operator fun invoke(vararg shapes: @SType("S") Int): @SType("S") Shape = Shape(shapes.clone())

        // TODO: Make this argument shape-typeable
        operator fun invoke(shapes: List<Int>): @SType("Shape") Shape = Shape(shapes.toIntArray())
    }
    init {
        assert(dims.size == rank)
        require(dims.all { it > 0 }) {
            "Cannot create a shape with dims $this because it contains a value <= 0"
        }
    }

    val rank get() = dims.size
    val first get() = dims[0]
    val last get() = dims[dims.size - 1]

    val product by lazy {
        dims.product
    }

    fun prepend(right: Int): Shape {
        val newRank = rank + 1
        val newData = IntArray(newRank) { if (it == 0) right else this[it - 1] }
        return Shape(newData)
    }

    @SType("B: Shape")
    @AllowUnreduced
    operator fun plus(right: @SType("B") Shape): @SType("concat(S,B)") Shape {
        if (this.isScalar) return right as @SType("concat(S,B)") Shape
        if (right.isScalar) return this
        val newRank = this.rank + right.rank
        val values = IntArray(newRank) { if (it < this.rank) this[it] else right[it - this.rank] }
        return Shape(values) as @SType("concat(S,B)") Shape
    }

    operator fun plus(right: Int): Shape {
        val newRank = rank + 1
        val newData = IntArray(newRank) { if (it < this.rank) this[it] else right }
        return Shape(newData)
    }

    fun reversed(): Shape = Shape(dims.reversedArray())

    fun product(): Int = product

    fun remove(axis: Int) = Shape(dims.remove(axis))

    val isScalar: Boolean get() = dims.isEmpty()

    val indices: IntRange get() = dims.indices

    fun take(n: Int): Shape {
        require(n in 0..rank)
        return when (n) {
            0 -> emptyShape
            rank -> this
            else -> Shape(dims.copyOfRange(0, n))
        }
    }

    fun drop(n: Int): Shape {
        require(n in 0..rank)
        return when (n) {
            0 -> this
            rank -> emptyShape
            else -> Shape(dims.copyOfRange(n, rank))
        }
    }

    fun dropLast(n: Int): Shape = take(rank - n)

    fun isPrefix(other: Shape): Boolean {
        if (this.rank > other.rank) return false
        for (i in 0 until this.rank)
            if (this[i] != other[i])
                return false
        return true
    }

    fun updated(axis: Int, newDim: Int): Shape {
        if (axis < 0 || axis >= rank)
            throw IndexOutOfBoundsException("index $axis out of bounds 0 until $rank")
        val newDims = IntArray(rank) { if (it == axis) newDim else dims[it] }
        return Shape(newDims)
    }

    operator fun get(idx: Int) = dims[idx]

    override fun hashCode(): Int {
        var result = Shape::class.hashCode() * 101 + rank
        for (d in dims) {
            result = result * 101 + d
        }
        return result
    }

    override fun equals(other: Any?): Boolean {
        if (other !is Shape || other.rank != this.rank)
            return false
        for (i in 0 until rank)
            if (dims[i] != other[i])
                return false
        return true
    }

    override fun toString() = "Shape(${ dims.joinToString { it.toString() }})"
}
