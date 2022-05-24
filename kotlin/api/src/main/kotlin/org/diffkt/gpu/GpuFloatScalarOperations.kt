/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.gpu

import org.diffkt.*
import org.diffkt.external.Gpu
import org.diffkt.model.BatchNormResult
import org.diffkt.random.RandomKey

internal object GpuFloatScalarOperations : Operations {
    override val name: String get() = "GpuFloatScalar"

    private fun wrap(value: DTensor): GpuFloatScalar {
        if (value is GpuFloatScalar) return value
        TODO("Cannot (automatically) convert to GpuFloatScalar")
    }

    override fun plus(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        val detachedAndResHandles = Gpu.add(l.handle, r.handle)
        // We explicitly wrap in a GpuFloatScalar for its side-effects
        // See the comment on [GpuFloatScalarOperations.plus].
        /* val detachedLhs = */ GpuFloatScalar(detachedAndResHandles[0])
        /* val detachedRhs = */ GpuFloatScalar(detachedAndResHandles[1])
        val res = GpuFloatScalar(detachedAndResHandles[2])
        return res
    }

    override fun minus(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        val resHandle = Gpu.sub(l.handle, r.handle)
        val res = GpuFloatScalar(resHandle)
        return res
    }

    override fun times(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        val l = wrap(left)
        val r = wrap(right)
        val resHandle = Gpu.times(l.handle, r.handle)
        val res = GpuFloatScalar(resHandle)
        return res
    }

    override fun timesScalar(left: DScalar, right: DTensor, derivativeId: DerivativeID): DTensor {
        TODO("Not yet implemented")
    }

    override fun zeroOfSameKind(x: DTensor, shape: Shape): DTensor {
        TODO("Not yet implemented")
    }

    override fun identityGradientOfSameKind(x: DTensor, halfShape: Shape): DTensor {
        TODO("Not yet implemented")
    }

    override fun unaryMinus(x: DTensor): DTensor {
        TODO("Not yet implemented")
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
        TODO("Not yet implemented")
    }

    override fun sin(x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun cos(x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun tan(x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun atan(x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun exp(x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun ln(x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun lgamma(x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun digamma(x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun polygamma(n: Int, x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun sqrt(x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun tanh(x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun meld(values: List<DTensor>, derivativeId: DerivativeID): DTensor {
        TODO("Not yet implemented")
    }

    override fun split(x: DTensor, shapes: List<Shape>): List<DTensor> {
        TODO("Not yet implemented")
    }

    override fun concat(left: DTensor, right: DTensor, axis: Int, derivativeId: DerivativeID): DTensor {
        throw IllegalArgumentException("DScalar lacks axis $axis")
    }

    override fun concat(slices: List<DTensor>, axis: Int, derivativeId: DerivativeID): DTensor {
        throw IllegalArgumentException("DScalar lacks axis $axis")
    }

    override fun broadcastTo(x: DTensor, newShape: Shape): DTensor {
        TODO("Not yet implemented")
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
        TODO("Not yet implemented")
    }

    override fun flip(x: DTensor, axes: IntArray): DTensor {
        TODO("Not yet implemented")
    }

    override fun logSoftmax(x: DTensor, axis: Int): DTensor {
        throw IllegalArgumentException("Cannot run logSoftmax on a scalar.")
    }

    override fun logSoftmaxGrad(x: DTensor, axis: Int, logSoftmax: DTensor, upstream: DTensor): DTensor {
        throw IllegalArgumentException("Cannot run logSoftmax on a scalar.")
    }

    override fun pow(base: DTensor, exponent: Float): DTensor {
        TODO("Not yet implemented")
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
        TODO("Not yet implemented")
    }

    override fun reshapeToScalar(x: DTensor): DScalar {
        throw IllegalStateException("scalar is already a scalar")
    }

    override fun squeeze(x: DTensor, axis: Int): DTensor {
        throw IllegalStateException("Cannot squeeze a scalar")
    }

    override fun unsqueeze(x: DTensor, axis: Int): DTensor {
        TODO("Not yet implemented")
    }

    override fun transpose(x: DTensor, axes: IntArray): DTensor {
        throw IllegalStateException("Cannot transpose a scalar")
    }

    override fun relu(x: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    override fun reluGrad(x: DTensor, reluUpstream: DTensor, derivativeId: DerivativeID): DTensor {
        TODO("Not yet implemented")
    }

    override fun sigmoid(x: DTensor): DTensor {
        TODO("Not yet implemented")
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

    override fun gamma(alpha: DTensor, randomKey: RandomKey): DTensor {
        TODO("Not yet implemented")
    }

    override fun compare(left: DTensor, right: DTensor, comparison: ComparisonKind): DTensor {
        TODO("Not yet implemented")
    }

    override fun ifThenElse(condition: DTensor, whenTrue: DTensor, whenFalse: DTensor, derivativeId: DerivativeID): DTensor {
        TODO("Not yet implemented")
    }
}