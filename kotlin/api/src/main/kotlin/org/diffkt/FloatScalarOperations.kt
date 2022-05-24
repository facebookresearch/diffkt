/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.external.Math
import org.diffkt.model.BatchNormResult
import org.diffkt.random.RandomKey
import org.diffkt.random.Sha512Random
import kotlin.math.*

internal object FloatScalarOperations: Operations {
    override val name get() = "FloatScalar"
    override val tensor: Operations get() = StridedFloatTensorOperations
    internal fun wrap(value: DTensor): FloatScalar {
        require(value.derivativeID.sequence == 0)
        require(value is FloatScalar)
        return value
    }

    override fun plus(left: DTensor, right: DTensor, derivativeId: DerivativeID): FloatScalar {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        return FloatScalar(l.value + r.value)
    }

    override fun minus(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        return FloatScalar(l.value - r.value)
    }

    override fun times(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        return FloatScalar(l.value * r.value)
    }

    override fun timesScalar(left: DScalar, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        return FloatScalar(l.value * r.value)
    }

    override fun div(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        return FloatScalar(l.value / r.value)
    }

    override fun zeroOfSameKind(x: DTensor, shape: Shape): DTensor {
        return if (shape.isScalar) FloatScalar.ZERO else FloatTensor.zeros(shape)
    }

    override fun identityGradientOfSameKind(x: DTensor, halfShape: Shape): DTensor {
        return StridedFloatTensorOperations.identityGradientOfSameKind(x, halfShape)
    }

    override fun unaryMinus(x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(-x.value)
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
        throw IllegalArgumentException("Matmul doesn't make sense for scalars")
    }

    override fun outerProduct(x: DTensor, y: DTensor, derivativeId: DerivativeID): DTensor {
        return x * y
    }

    override fun sin(x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(sin(x.value))
    }

    override fun cos(x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(cos(x.value))
    }

    override fun tan(x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(tan(x.value))
    }

    override fun atan(x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(atan(x.value))
    }

    override fun exp(x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(exp(x.value))
    }

    override fun ln(x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(ln(x.value))
    }

    override fun lgamma(x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(Math.lgamma(x.value))
    }

    override fun digamma(x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(Math.digamma(x.value))
    }

    override fun polygamma(n: Int, x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(Math.polygamma(n, x.value))
    }

    override fun sqrt(x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(sqrt(x.value))
    }

    override fun tanh(x: DTensor): DTensor {
        x as FloatScalar
        return FloatScalar(tanh(x.value))
    }

    override fun meld(values: List<DTensor>, derivativeId: DerivativeID): DTensor {
        return StridedFloatTensorOperations.meld(values, derivativeId)
    }

    override fun split(x: DTensor, shapes: List<Shape>): List<DTensor> {
        return StridedFloatTensorOperations.split(x, shapes)
    }

    override fun concat(left: DTensor, right: DTensor, axis: Int, derivativeId: DerivativeID): DTensor {
        throw IllegalArgumentException("DScalar lacks axis $axis")
    }

    override fun concat(slices: List<DTensor>, axis: Int, derivativeId: DerivativeID): DTensor {
        throw IllegalArgumentException("DScalar lacks axis $axis")
    }

    override fun broadcastTo(x: DTensor, newShape: Shape): DTensor {
        x as FloatScalar
        return StridedFloatTensor.singleton(newShape, x.value)
    }

    override fun convImpl(
        signal: DTensor,
        filter: DTensor,
        hStride: Int,
        vStride: Int,
        padding: Convolve.Padding2D,
        derivativeId: DerivativeID
    ): DTensor {
        throw IllegalArgumentException("Cannot run convolution on a scalar.")
    }

    override fun expand(x: DTensor, newShape: Shape): DTensor {
        return x
    }

    override fun flip(x: DTensor, axes: IntArray): DTensor {
        return x
    }

    override fun logSoftmax(x: DTensor, axis: Int): DTensor {
        throw IllegalArgumentException("Cannot run logSoftmax on a scalar.")
    }

    override fun logSoftmaxGrad(x: DTensor, axis: Int, logSoftmax: DTensor, upstream: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun pow(base: DTensor, exponent: Float): DTensor {
        base as FloatScalar
        return FloatScalar(base.value.pow(exponent))
    }

    override fun view1(x: DTensor, indices: IntArray): DTensor {
        return x
    }

    override fun view2(x: DTensor, index: Int, axis: Int): DTensor {
        throw IllegalArgumentException("view not applicable to scalar")
    }

    override fun view3(x: DTensor, index: IntRange, axis: Int): DTensor {
        throw IllegalArgumentException("view not applicable to scalar")
    }

    override fun reshape(x: DTensor, newShape: Shape): DTensor {
        require(x is FloatScalar)
        return StridedFloatTensor(newShape, offset = 0, strides = StridedUtils.singletonStrides(newShape.rank), floatArrayOf(x.value), StridedUtils.Layout.SINGLETON)
    }

    override fun reshapeToScalar(x: DTensor): DScalar {
        throw IllegalStateException("scalar is already a scalar")
    }

    override fun squeeze(x: DTensor, axis: Int): DTensor {
        throw IllegalStateException("Cannot squeeze a scalar")
    }

    override fun unsqueeze(x: DTensor, axis: Int): DTensor {
        require(x is DScalar)
        return tensorOf(x)
    }

    override fun transpose(x: DTensor, axes: IntArray): DTensor {
        throw IllegalStateException("Cannot transpose a scalar")
    }

    override fun relu(x: DTensor): DTensor {
        require(x is FloatScalar)
        return if (x.value <= 0f) FloatScalar.ZERO else x
    }

    override fun reluGrad(x: DTensor, reluUpstream: DTensor, derivativeId: DerivativeID): DTensor {
        require(x is FloatScalar)
        return if (x.value <= 0f) reluUpstream.operations.zeroOfSameKind(reluUpstream, reluUpstream.shape) else  reluUpstream
    }

    override fun sigmoid(x: DTensor): DTensor {
        require(x is FloatScalar)
        return FloatScalar(sigmoidElem(x.value))
    }

    override fun sum(x: DTensor, axes: IntArray, keepDims: Boolean): DTensor {
        throw IllegalStateException("Cannot sum a scalar")
    }

    override fun avgPool(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        throw IllegalStateException("Cannot perform avgPool on a scalar")
    }

    override fun avgPoolGrad(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        throw IllegalStateException("Cannot perform avgPoolGrad on a scalar")
    }

    override fun batchNorm(input: DTensor, scaleShift: DTensor, derivativeId: DerivativeID): BatchNormResult {
        throw IllegalStateException("Cannot perform batchNorm on a scalar")
    }

    override fun maxPoolWithIndices(
        x: DTensor,
        poolHeight: Int,
        poolWidth: Int,
        withIndices: Boolean
    ): Pair<DTensor, List<IntArray>?> {
        throw IllegalStateException("Cannot perform maxPoolWithIndices on a scalar")
    }

    override fun gather(x: DTensor, indices: List<Int>, axis: Int, paddingIndex: Int): DTensor {
        throw IllegalStateException("Cannot perform gather on a scalar")
    }

    override fun gatherAtIndices(x: DTensor, indices: List<IntArray>): DTensor {
        throw IllegalStateException("Cannot perform gatherAtIndices on a scalar")
    }

    override fun scatter(x: DTensor, indices: List<Int>, axis: Int, newShape: Shape, paddingIndex: Int): DTensor {
        throw IllegalStateException("Cannot perform scatter on a scalar")
    }

    override fun scatterAtIndices(x: DTensor, indices: List<IntArray>, newShape: Shape): DTensor {
        throw IllegalStateException("Cannot perform scatterAtIndices on a scalar")
    }

    override fun gamma(alpha: DTensor, randomKey: RandomKey): DScalar {
        require(alpha is FloatScalar)

        return (randomKey as Sha512Random).gamma(alpha) as FloatScalar
    }

    override fun compare(left: DTensor, right: DTensor, comparison: ComparisonKind): DTensor {
        require(left is FloatScalar)
        require(right is FloatScalar)
        val satisfied = compare(left.value, right.value, comparison)
        return if (satisfied) FloatScalar.ONE else FloatScalar.ZERO
    }

    override fun ifThenElse(condition: DTensor, whenTrue: DTensor, whenFalse: DTensor, derivativeId: DerivativeID): DTensor {
        require(condition is FloatScalar)
        return if (condition.value > 0f) whenTrue else whenFalse
    }
}
