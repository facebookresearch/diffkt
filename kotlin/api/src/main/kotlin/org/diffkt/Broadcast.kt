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
internal fun @SType("Shape") DTensor.broadcastTo(newShape: @SType("S") Shape): @SType("S") DTensor {
    require(this.rank <= newShape.rank)
    if (shape == newShape) return this
    return this.operations.broadcastTo(this, newShape)
}

internal object Broadcasting {
    /** Returns arguments broadcasted to a common shape. */
    fun broadcastToCommonShape(a: DTensor, b: DTensor): Pair<DTensor, DTensor> {
        if (a.shape == b.shape) return Pair(a, b)
        val newShape = broadcastShape(a.shape, b.shape)
        return Pair(a.broadcastTo(newShape), b.broadcastTo(newShape))
    }

    /** Returns arguments broadcasted to a common shape. */
    fun broadcastToCommonShape(a: DTensor, b: DTensor, c: DTensor): Triple<DTensor, DTensor, DTensor> {
        if (a.shape == b.shape && a.shape == c.shape) return Triple(a, b, c)
        val newShape = broadcastShapeImpl(a.shape, b.shape, c.shape)
        return Triple(a.broadcastTo(newShape), b.broadcastTo(newShape), c.broadcastTo(newShape))
    }

    fun getBroadcastedAxes(shape: Shape, newShape: Shape) =
        getBroadcastedStridesAndAxes(shape, IntArray(shape.rank) { 0 }, newShape).second

    /**
     * Given a Tensor and a shape to broadcast to, return the strides for the newly
     * broadcasted tensor and list of modified axes (for calculating the gradient)
     */
    fun getBroadcastedStridesAndAxes(shape: Shape, strides: IntArray, newShape: Shape): Pair<IntArray, IntArray> {
        return if (shape == newShape) {
            Pair(strides, IntArray(0))
        } else {
            val xr = shape.rank
            val r = newShape.rank
            if (xr > r)
                throw IllegalArgumentException("broadcastTo: tensor rank $xr > broadcast rank $r")
            val rdiff = r - xr
            // shape.take(rdiff) concat shape
            val shapeScratchpad = MutableList(r) { if (it < rdiff) newShape[it] else shape[it - rdiff] }
            // zeros(rdiff) concat strides
            val stridesScratchpad = MutableList(r) { if (it < rdiff) 0 else strides[it - rdiff] }
            val axes = mutableListOf<Int>()

            for (i in 0 until r) {
                val dim = shapeScratchpad[i]
                val newDim = newShape[i]
                if (dim != newDim) {
                    if (dim == 1) {
                        shapeScratchpad[i] = newDim
                        stridesScratchpad[i] = 0
                        axes += i
                    } else {
                        assert(newDim != 1) { "cannot broadcast dim $i from $dim to 1" }
                        if (newDim != 1) {
                            throw java.lang.RuntimeException("broadcast: incompatible shapes at dim $i: ${shape} $newShape")
                        }
                    }
                }
            }
            assert(Shape(shapeScratchpad.toIntArray()) == newShape)
            Pair(stridesScratchpad.toIntArray(), axes.toIntArray())
        }
    }

    fun broadcastShape(xShape: Shape, yShape: Shape): Shape =
        broadcastShapeImpl(xShape, yShape)

    private fun broadcastShapeImpl(vararg shapes: Shape): Shape {
        if (shapes.isEmpty())
            throw IllegalArgumentException("broadcastShape called with no tensors")
        var shape = MutableList(shapes[0].rank) { shapes[0][it] }
        for (i in 1..shapes.lastIndex) {
            val currShape = shapes[i]
            var rdiff = shape.size - currShape.rank
            if (rdiff < 0) {
                shape = (currShape.take(-rdiff) + Shape(shape.toIntArray())).dims.toMutableList()
                rdiff = 0
            }
            for (j in rdiff..shape.lastIndex) {
                val sd = shape[j]
                val td = currShape[j - rdiff]
                if (sd == 1)
                    shape[j] = td
                else if (td != 1 && td != sd)
                    throw RuntimeException("broadcast: incompatible shapes at dim $j: $sd and $td")
            }
        }
        return Shape(shape.toIntArray())
    }
}
