/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import org.diffkt.*
import org.diffkt.random.RandomKey
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

@AllowUnreduced
object TracingTensorOperations : Operations {
    override val name: String
        get() = "TracingTensor"

    fun wrap(x: DTensor)= TracingTensor.wrap(x)

    @SType("S: Shape")
    override fun plus(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        val l = wrap(left)
        val r = wrap(right)
        return if (l is TracingScalar)
            TracingScalar.Plus(l, r as TracingScalar)
        else
            TracingTensor.Plus(l, r)
    }

    @SType("S: Shape")
    override fun minus(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        val l = wrap(left)
        val r = wrap(right)
        return if (l is TracingScalar && r is TracingScalar)
            TracingScalar.Minus(l, r)
        else
            TracingTensor.Minus(l, r)
    }

    @SType("S: Shape")
    override fun times(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(right !is DScalar)
        val l = wrap(left)
        val r = wrap(right)
        return TracingTensor.Times(l, r)
    }

    @SType("S: Shape")
    override fun timesScalar(left: DScalar, right: @SType("S") DTensor, derivativeId: DerivativeID): @SType("S") DTensor {
        val l = wrap(left) as TracingScalar
        val r = wrap(right)
        return if (r is TracingScalar)
            TracingScalar.TimesScalar(l, r)
        else
            TracingTensor.TimesScalar(l, r)
    }

    @SType("S: Shape")
    override fun div(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        val l = wrap(left)
        val r = wrap(right)
        return if (l is TracingScalar)
            TracingScalar.Div(l, r as TracingScalar)
        else
            TracingTensor.Div(l, r)
    }

    @SType("S: Shape")
    override fun zeroOfSameKind(x: DTensor, shape: @SType("S") Shape): @SType("S") DTensor {
        return when {
            shape.product < 10 -> StridedFloatTensorOperations.zeroOfSameKind(x, shape)
            // shape.isScalar -> TracingScalar.Zero()
            else -> TracingTensor.Zero(shape)
        }
    }

    @SType("S: Shape")
    override fun identityGradientOfSameKind(x: DTensor, halfShape: @SType("S") Shape): @SType("concat(S,S)") DTensor {
        return when {
            halfShape.product < 4 -> StridedFloatTensorOperations.identityGradientOfSameKind(x, halfShape)
            // halfShape.isScalar -> TracingScalar.Constant(FloatScalar(1F))
            else -> TracingTensor.IdentityGradient(halfShape)
        }
    }

    @SType("S: Shape")
    override fun unaryMinus(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.UnaryMinus(xx)
        else
            TracingTensor.UnaryMinus(xx)
    }

    override fun matmul(
        x: DTensor,
        y: DTensor,
        a: Shape,
        b: Shape,
        c: Shape,
        d: Shape,
        derivativeId: DerivativeID
    ): DTensor {
        val xx = wrap(x)
        val yy = wrap(y)
        return if (a.isScalar && b.isScalar && d.isScalar)
            TracingScalar.Matmul(xx, yy, a, b, c, d)
        else
            TracingTensor.Matmul(xx, yy, a, b, c, d)
    }

    @SType("S1: Shape, S2: Shape")
    override fun outerProduct(
        x: @SType("S1") DTensor,
        y: @SType("S2") DTensor,
        derivativeId: DerivativeID
    ): @SType("concat(S1, S2)") DTensor {
        val xx = wrap(x)
        val yy = wrap(y)
        require(!(xx.shape + yy.shape).isScalar)
        return TracingTensor.OuterProduct(xx, yy)
    }

    @SType("S: Shape")
    override fun sin(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Sin(xx)
        else
            TracingTensor.Sin(xx)
    }

    @SType("S: Shape")
    override fun cos(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Cos(xx)
        else
            TracingTensor.Cos(xx)
    }

    @SType("S: Shape")
    override fun tan(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Tan(xx)
        else
            TracingTensor.Tan(xx)
    }

    @SType("S: Shape")
    override fun atan(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Atan(xx)
        else
            TracingTensor.Atan(xx)
    }

    @SType("S: Shape")
    override fun exp(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Exp(xx)
        else
            TracingTensor.Exp(xx)
    }

    @SType("S: Shape")
    override fun ln(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Ln(xx)
        else
            TracingTensor.Ln(xx)
    }

    @SType("S: Shape")
    override fun lgamma(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Lgamma(xx)
        else
            TracingTensor.Lgamma(xx)
    }

    @SType("S: Shape")
    override fun digamma(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Digamma(xx)
        else
            TracingTensor.Digamma(xx)
    }

    @SType("S: Shape")
    override fun polygamma(n: Int, x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Polygamma(n, xx)
        else
            TracingTensor.Polygamma(n, xx)
    }

    @SType("S: Shape")
    override fun sqrt(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Sqrt(xx)
        else
            TracingTensor.Sqrt(xx)
    }

    @SType("S: Shape")
    override fun tanh(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Tanh(xx)
        else
            TracingTensor.Tanh(xx)
    }

    override fun meld(values: List<DTensor>, derivativeId: DerivativeID): DTensor {
        val wrappedValues = values.map { wrap(it) }
        return TracingTensor.Meld(wrappedValues)
    }

    override fun split(x: DTensor, shapes: List<Shape>): List<DTensor> {
        val xx = wrap(x)
        val split = if (xx is TracingScalar)
            TracingScalar.Split(xx, shapes)
        else
            TracingTensor.Split(xx, shapes)
        return split.splitValues
    }

    @SType("S1: Shape, S2: Shape, A: Dim")
    override fun concat(
        left: @SType("S1")  DTensor,
        right: @SType("S2") DTensor,
        axis: @SType("A") Int,
        derivativeId: DerivativeID
    ): @SType("concatOnAxis(S1, S2, A)") DTensor {
        val l = wrap(left)
        val r = wrap(right)
        return concat(listOf(l, r), axis, derivativeId)
    }

    override fun concat(slices: List<DTensor>, axis: Int, derivativeId: DerivativeID): DTensor {
        return TracingTensor.Concat(slices.map { wrap(it) }, axis)
    }

    override fun broadcastTo(x: DTensor, newShape: Shape): DTensor {
        val xx = wrap(x)
        require(!newShape.isScalar)
        return TracingTensor.BroadcastTo(xx, newShape)
    }

    override fun convImpl(
        signal: DTensor,
        filter: DTensor,
        hStride: Int,
        vStride: Int,
        padding: Convolve.Padding2D,
        derivativeId: DerivativeID
    ): DTensor {
        val s = wrap(signal)
        val f = wrap(filter)
        return TracingTensor.ConvImpl(s, f, hStride, vStride, padding)
    }

    override fun expand(x: DTensor, newShape: Shape): DTensor {
        val xx = wrap(x)
        require(!newShape.isScalar)
        return TracingTensor.Expand(xx, newShape)
    }

    @SType("S: Shape")
    override fun flip(x: @SType("S") DTensor, axes: IntArray): @SType("S") DTensor {
        val xx = wrap(x)
        return TracingTensor.Flip(xx, axes)
    }

    override fun logSoftmax(x: DTensor, axis: Int): DTensor {
        val xx = wrap(x)
        return TracingTensor.LogSoftmax(xx, axis)
    }

    override fun logSoftmaxGrad(x: DTensor, axis: Int, logSoftmax: DTensor, upstream: DTensor): DTensor {
        val xx = wrap(x)
        val l = wrap(logSoftmax)
        val u = wrap(upstream)
        return TracingTensor.LogSoftmaxGrad(xx, axis, l, u)
    }

    @SType("S: Shape")
    override fun pow(base: @SType("S") DTensor, exponent: Float): @SType("S") DTensor {
        val b = wrap(base)
        return if (b is TracingScalar) TracingScalar.Pow(b, exponent) else TracingTensor.Pow(b, exponent)
    }

    override fun view1(x: DTensor, indices: IntArray): DTensor {
        val xx = wrap(x)
        val newShape = xx.shape.drop(indices.size)
        return if (newShape.isScalar) TracingScalar.View1(xx, indices) else TracingTensor.View1(xx, indices, newShape)
    }

    override fun view2(x: DTensor, index: Int, axis: Int): DTensor {
        val xx = wrap(x)
        val newShape = xx.shape.remove(axis)
        return if (newShape.isScalar) TracingScalar.View2(xx, index, axis) else TracingTensor.View2(xx, index, axis, newShape)
    }

    override fun view3(x: DTensor, index: IntRange, axis: Int): DTensor {
        val xx = wrap(x)
        return TracingTensor.View3(xx, index, axis)
    }

    override fun reshape(x: DTensor, newShape: Shape): DTensor {
        val xx = wrap(x)
        require(!newShape.isScalar)
        return TracingTensor.Reshape(xx, newShape)
    }

    override fun reshapeToScalar(x: DTensor): DScalar {
        val xx = wrap(x)
        return TracingScalar.ReshapeToScalar(xx)
    }

    override fun squeeze(x: DTensor, axis: Int): DTensor {
        val xx = wrap(x)
        return if (xx.rank == 1) TracingScalar.Squeeze(xx) else TracingTensor.Squeeze(xx, axis)
    }

    override fun unsqueeze(x: DTensor, axis: Int): DTensor {
        val xx = wrap(x)
        return TracingTensor.Unsqueeze(xx, axis)
    }

    override fun transpose(x: DTensor, axes: IntArray): DTensor {
        val xx = wrap(x)
        return TracingTensor.Transpose(xx, axes)
    }

    @SType("S: Shape")
    override fun relu(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Relu(xx)
        else
            TracingTensor.Relu(xx)
    }

    override fun reluGrad(x: DTensor, reluUpstream: DTensor, derivativeId: DerivativeID): DTensor {
        val xx = wrap(x)
        val up = wrap(reluUpstream)
        return if (xx is TracingScalar)
            TracingScalar.ReluGrad(xx, up)
        else
            TracingTensor.ReluGrad(xx, up)
    }

    @SType("S: Shape")
    override fun sigmoid(x: @SType("S") DTensor): @SType("S") DTensor {
        val xx = wrap(x)
        return if (xx is TracingScalar)
            TracingScalar.Sigmoid(xx)
        else
            TracingTensor.Sigmoid(xx)
    }

    override fun sum(x: DTensor, axes: IntArray, keepDims: Boolean): DTensor {
        val xx = wrap(x)
        return if (!keepDims && xx.rank == axes.size)
            TracingScalar.Sum(xx)
        else
            TracingTensor.Sum(xx, axes, keepDims)
    }

    override fun avgPool(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        val xx = wrap(x)
        return TracingTensor.AvgPool(xx, poolHeight, poolWidth)
    }

    override fun avgPoolGrad(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        val xx = wrap(x)
        return TracingTensor.AvgPoolGrad(xx, poolHeight, poolWidth)
    }

    override fun maxPoolWithIndices(
        x: DTensor,
        poolHeight: Int,
        poolWidth: Int,
        withIndices: Boolean
    ): Pair<DTensor, List<IntArray>?> {
        val xx = wrap(x)
        require(!withIndices) { "TracingTensor: maxPool not supported with indices" }
        val result = TracingTensor.MaxPoolWithIndices(xx, poolHeight, poolWidth, withIndices)
        return Pair(result, null)
    }

    override fun gather(x: DTensor, indices: List<Int>, axis: Int, paddingIndex: Int): DTensor {
        val xx = wrap(x)
        return TracingTensor.Gather(xx, indices, axis, paddingIndex)
    }

    override fun gatherAtIndices(x: DTensor, indices: List<IntArray>): DTensor {
        val xx = wrap(x)
        return TracingTensor.GatherAtIndices(xx, indices)
    }

    override fun scatter(x: DTensor, indices: List<Int>, axis: Int, newShape: Shape, paddingIndex: Int): DTensor {
        val xx = wrap(x)
        return TracingTensor.Scatter(xx, indices, axis, newShape, paddingIndex)
    }

    override fun scatterAtIndices(x: DTensor, indices: List<IntArray>, newShape: Shape): DTensor {
        val xx = wrap(x)
        return TracingTensor.ScatterAtIndices(xx, indices, newShape)
    }

    override fun gamma(alpha: DTensor, randomKey: RandomKey): DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun compare(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        comparison: ComparisonKind
    ): @SType("S") DTensor {
        require(left.isScalar == right.isScalar)
        val l = wrap(left)
        val r = wrap(right)
        return if (l is TracingScalar && r is TracingScalar)
            TracingScalar.Compare(l, r, comparison)
        else
            TracingTensor.Compare(l, r, comparison)
    }

    @SType("S: Shape")
    override fun ifThenElse(
        condition: @SType("S") DTensor,
        whenTrue: @SType("S") DTensor,
        whenFalse: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        val cond = wrap(condition)
        val t = wrap(whenTrue)
        val f = wrap(whenFalse)
        return if (t is TracingScalar && f is TracingScalar)
            TracingScalar.IfThenElse(cond, t, f)
        else
            TracingTensor.IfThenElse(cond, t, f)
    }
}
