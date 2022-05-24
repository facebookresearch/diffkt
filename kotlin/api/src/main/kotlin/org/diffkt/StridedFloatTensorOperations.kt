/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.external.Dnnl
import org.diffkt.external.External
import org.diffkt.external.Math
import org.diffkt.external.Predicate
import org.diffkt.model.BatchNormResult
import org.diffkt.model.maxPoolWithIndicesDnnl
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

@AllowUnreduced
internal object StridedFloatTensorOperations : FloatTensorOperations() {
    override val name get() = "StridedFloatTensor"

    private fun wrap(value: DTensor): StridedFloatTensor {
        require(value.derivativeID.sequence == 0)
        require(value is FloatTensor)
        return value.asStrided()
    }

    @SType("S: Shape")
    override fun plus(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        return if (shouldSendToCpp(200, Dnnl, l, r, checkLayout = false, checkOffset = false))
            Dnnl.add(l, r)
        else
            super.plus(left, right, derivativeId)
    }

    @SType("S: Shape")
    override fun minus(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        return if (shouldSendToCpp(200, Dnnl, l, r, checkLayout = false, checkOffset = false))
            Dnnl.sub(l, r)
        else
            super.minus(left, right, derivativeId)
    }

    @SType("S: Shape")
    override fun times(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        val l = wrap(left)
        val r = wrap(right)
        return if (shouldSendToCpp(200, External, l, r))
            FloatTensor(left.shape, Math.times(l.data, r.data, left.size))
        else
            super.times(left, right, derivativeId)
    }

    @SType("S: Shape")
    override fun timesScalar(left: DScalar, right: @SType("S") DTensor, derivativeId: DerivativeID): @SType("S") DTensor {
        val r = wrap(right)
        val leftValue = FloatScalarOperations.wrap(left).value
        return if (shouldSendToCpp(32, Dnnl, r, checkLayout = false))
            Dnnl.mulScalar(r, leftValue)
        else
            super.timesScalar(left, right, derivativeId)
    }

    @SType("S: Shape")
    override fun unaryMinus(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is StridedFloatTensor)
        return if (shouldSendToCpp(100, External, x))
            FloatTensor(x.shape, Math.unaryMinus(x.data, x.size))
        else
            super.unaryMinus(x)
    }

    override fun view1(x: DTensor, indices: IntArray): DTensor {
        require(x is StridedFloatTensor)
        val newShape = x.shape.drop(indices.size)
        val newOffset = indices.indices.fold(x.offset) { o, i -> o + x.strides[i] * indices[i] }
        return if (newShape.isScalar)
            FloatScalar(x.data[newOffset])
        else {
            val newStrides = x.strides.sliceArray(indices.size until x.shape.rank)
            StridedFloatTensor(newShape, newOffset, newStrides, x.data)
        }
    }

    override fun view2(x: DTensor, index: Int, axis: Int): DTensor {
        require(x is StridedFloatTensor)
        val newShape = x.shape.remove(axis)
        return if (newShape.isScalar)
            FloatScalar(x.at(index))
        else
            StridedFloatTensor(newShape, x.offset + index * x.strides[axis], x.strides.remove(axis), x.data)
    }

    override fun view3(x: DTensor, index: IntRange, axis: Int): DTensor {
        require(x is StridedFloatTensor)
        val dimSize = 1 + (index.endInclusive - index.start) / index.step
        val newShape = x.shape.updated(axis, dimSize)
        val newOffset = x.offset + index.start * x.strides[axis]
        val newStrides = IntArray(x.strides.size) { if (it == axis) x.strides[axis] * index.step else x.strides[it] }
        return StridedFloatTensor(newShape, newOffset, newStrides, x.data)
    }

    override fun reshape(x: DTensor, newShape: Shape): DTensor {
        require(x is StridedFloatTensor)
        return when (x.layout) {
            StridedUtils.Layout.SINGLETON ->
                StridedFloatTensor(newShape, x.offset, strides = StridedUtils.singletonStrides(newShape.rank), x.data, x.layout)
            StridedUtils.Layout.NATURAL ->
                StridedFloatTensor(newShape, x.offset, strides = StridedUtils.contigStrides(newShape), x.data, x.layout)
            else -> reshape(x.normalize(), newShape)
        }
    }

    override fun squeeze(x: DTensor, axis: Int): DTensor {
        require(x is StridedFloatTensor)
        val str = x.strides
        val sh = x.shape
        val shape = sh.take(axis) + sh.drop(axis + 1)
        val strides = str.take(axis) + str.drop(axis + 1)
        return StridedFloatTensor(shape, x.offset, strides.toIntArray(), x.data)
    }

    override fun unsqueeze(x: DTensor, axis: Int): DTensor {
        require(x is StridedFloatTensor)
        val str = x.strides
        val sh = x.shape
        val unstride = when (axis) {
            str.size -> 1
            0 -> str[0] * sh[0]
            else -> str[axis - 1]
        }
        val shape = sh.take(axis) + 1 + sh.drop(axis)
        val strides = (str.take(axis) + unstride + str.drop(axis)).toIntArray()
        return StridedFloatTensor(shape, x.offset, strides, x.data)
    }

    override fun transpose(x: DTensor, axes: IntArray): DTensor {
        require(x is StridedFloatTensor)
        fun shuffle(a: IntArray): IntArray {
            return axes.map { i -> a[i] }.toIntArray()
        }
        val transposedShape = Shape(shuffle(x.shape.dims))
        val transposedStrides = shuffle(x.strides)
        return StridedFloatTensor(transposedShape, x.offset, transposedStrides, x.data)
    }

    @SType("S: Shape")
    override fun relu(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is StridedFloatTensor)
        // TODO: set a not-arbitrary threshold
        return if (shouldSendToCpp(10, x, Dnnl, checkLayout = false)) {
            val normalized = x.normalize()
            StridedFloatTensor.contiguous(x.shape) { data ->
                Dnnl.relu(x.shape.dims, data, normalized.data)
            }
        } else {
            super.relu(x)
        }
    }

    override fun reluGrad(x: DTensor, reluUpstream: DTensor, derivativeId: DerivativeID): DTensor {
        val xx = wrap(x)
        val up = wrap(reluUpstream)
        return when {
            reluUpstream.shape == x.shape && shouldSendToCpp(10, Dnnl, xx, up, checkLayout = true) -> {
                StridedFloatTensor.contiguous(x.shape) { data ->
                    Dnnl.reluGrad(x.shape.dims, data, up.data, xx.data)
                }
            }
            else -> super.reluGrad(x, reluUpstream, derivativeId)
        }
    }

    override fun sum(x: DTensor, axes: IntArray, keepDims: Boolean): DTensor {
        require(x is StridedFloatTensor)
        // TODO: pick a non-arbitrary size threshold
        return if (shouldSendToCpp(100, Dnnl, x, checkLayout = false)) {
            fun resShape(keepDims: Boolean) = if (keepDims)
                Shape(x.shape.dims.mapIndexed { ix, it -> if (ix in axes) 1 else it }.toIntArray())
            else
                Shape(x.shape.dims.filterIndexed { ix, _ -> ix !in axes }.toIntArray())

            val resultShape = resShape(keepDims)
            val resultShapeForDnnl = resShape(true)
            val resultData = FloatArray(resultShape.product)
            // TODO: handle non contig data efficiently
            Dnnl.reduceSum(resultShapeForDnnl.dims, resultData, x.shape.dims, x.normalize().data)
            FloatTensor(resultShape, resultData)
        } else {
            super.sum(x, axes, keepDims)
        }
    }

    override fun batchNorm(input: DTensor, scaleShift: DTensor, derivativeId: DerivativeID): BatchNormResult {
        return if (input.rank == 4
            && isDnnlEligible(input)
            && isDnnlEligible(scaleShift)) {
            input as FloatTensor
            scaleShift as FloatTensor
            val C = input.shape[3]
            require(scaleShift.shape == Shape(2, C)) { "scaleShift must have shape ${Shape(2, C)}" }

            val result = StridedFloatTensor.contigZeros(input.shape)
            val mean = StridedFloatTensor.contigZeros(Shape(C))
            val variance = StridedFloatTensor.contigZeros(Shape(C))

            Dnnl.batchNorm(result.shape.dims, result.data, mean.data, variance.data,
                input.normalize().data, scaleShift.normalize().data)
            BatchNormResult.fromMeanAndVariance(result, mean, variance)
        }
        else {
            super.batchNorm(input, scaleShift, derivativeId)
        }
    }

    override fun maxPoolWithIndices(
        x: DTensor,
        poolHeight: Int,
        poolWidth: Int,
        withIndices: Boolean
    ): Pair<DTensor, List<IntArray>?> {
        require(x is StridedFloatTensor)
        return if (shouldSendToCpp(100, Dnnl, x) && !withIndices)
            Pair(maxPoolWithIndicesDnnl(x, poolHeight, poolWidth, withIndices = false).first, null)
        else
            super.maxPoolWithIndices(x, poolHeight, poolWidth, withIndices)
    }

    @SType("S: Shape")
    override fun ifThenElse(
        condition: @SType("S") DTensor,
        whenTrue: @SType("S") DTensor,
        whenFalse: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        condition as StridedFloatTensor
        whenTrue as StridedFloatTensor
        whenFalse as StridedFloatTensor
        return if (shouldSendToCpp(200, External, condition, whenTrue, whenFalse))
            FloatTensor(whenTrue.shape, Predicate.ifThenElse(
                condition.data,
                whenTrue.data,
                whenFalse.data,
                condition.size,
            ))
        else
            super.ifThenElse(condition, whenTrue, whenFalse, derivativeId)
    }
}
