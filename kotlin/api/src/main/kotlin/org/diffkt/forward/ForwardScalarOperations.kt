/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.forward

import org.diffkt.*
import org.diffkt.model.BatchNormResult
import org.diffkt.random.RandomKey
import org.diffkt.tracing.TracingScalar
import shapeTyping.annotations.AllowUnreduced

internal object ForwardScalarOperations: Operations {
    override val name get() = "ForwardScalar"
    override val tensor: Operations get() = ForwardTensorOperations
    internal fun wrap(value: DTensor, derivativeId: ForwardDerivativeID): ForwardScalar {
        require(value is DScalar)
        if (value is ForwardScalar && value.derivativeID == derivativeId)
            return value
        require(value.derivativeID.sequence < derivativeId.sequence)
        // TODO: the zeroes, below, should be of an appropriate species of tensor, not necessarily a float tensor
        return ForwardScalar(value, derivativeId, FloatTensor.zeros(value.shape + derivativeId.inputTangentShapeForJacobian))
    }

    override fun plus(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ForwardDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return ForwardScalar(l.primal + r.primal, derivativeId, l.tangent + r.tangent)
    }

    override fun minus(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ForwardDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return ForwardScalar(l.primal - r.primal, derivativeId, l.tangent - r.tangent)
    }

    override fun times(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ForwardDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        val primal = l.primal * r.primal
        val tangent = l.primal * r.tangent + r.primal * l.tangent
        return ForwardTensor(primal, derivativeId, tangent)
    }

    override fun timesScalar(left: DScalar, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ForwardDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        val primal = l.primal * r.primal
        val tangent = l.primal * r.tangent + r.primal * l.tangent
        return ForwardTensor(primal, derivativeId, tangent)
    }

    override fun div(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ForwardDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        val primal = l.primal / r.primal
        val tangent = (l.tangent * r.primal - l.primal * r.tangent) / (r.primal * r.primal)
        return ForwardTensor(primal, derivativeId, tangent)
    }

    override fun zeroOfSameKind(x: DTensor, shape: Shape): DTensor {
        require(x is ForwardScalar)
        val primal = x.primal.operations.zeroOfSameKind(x.primal, shape) as DScalar
        val tangent = x.tangent.operations.zeroOfSameKind(x.tangent, shape + x.derivativeID.inputTangentShapeForJacobian)
        return ForwardScalar(primal, x.derivativeID, tangent)
    }

    @AllowUnreduced
    override fun identityGradientOfSameKind(x: DTensor, halfShape: Shape): DTensor {
        return x.primal.operations.identityGradientOfSameKind(x.primal, halfShape)
    }

    override fun unaryMinus(x: DTensor): DTensor {
        require(x is ForwardScalar)
        return ForwardScalar(-x.primal, x.derivativeID, -x.tangent)
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
        require(x is ForwardScalar)
        return ForwardScalar(
            org.diffkt.sin(x.primal),
            x.derivativeID,
            x.tangent * org.diffkt.cos(x.primal)
        )
    }

    override fun cos(x: DTensor): DTensor {
        require(x is ForwardScalar)
        return ForwardScalar(
            org.diffkt.cos(x.primal),
            x.derivativeID,
            x.tangent * org.diffkt.sin(-x.primal)
        )
    }

    override fun tan(x: DTensor): DTensor {
        require(x is ForwardScalar)
        val primal = org.diffkt.tan(x.primal)
        val cos = org.diffkt.cos(x.primal)
        val tangent = x.tangent / (cos * cos)
        return ForwardScalar(primal, x.derivativeID, tangent)
    }

    override fun atan(x: DTensor): DTensor {
        require(x is ForwardScalar)
        val primal = org.diffkt.atan(x.primal)
        val tangent = x.tangent / (1f + x.primal.pow(2))
        return ForwardScalar(primal, x.derivativeID, tangent)
    }

    override fun exp(x: DTensor): DTensor {
        require(x is ForwardScalar)
        return org.diffkt.exp(x.primal)
            .let { ForwardScalar(it, x.derivativeID, x.tangent * it) }
    }

    override fun ln(x: DTensor): DTensor {
        require(x is ForwardScalar)
        return ForwardScalar(
            org.diffkt.ln(x.primal),
            x.derivativeID,
            x.primal.pow(-1f) * x.tangent
        )
    }

    override fun lgamma(x: DTensor): DTensor {
        require(x is ForwardScalar)
        return ForwardScalar(
            org.diffkt.lgamma(x.primal),
            x.derivativeID,
            org.diffkt.digamma(x.primal) * x.tangent
        )
    }

    override fun digamma(x: DTensor): DTensor {
        require(x is ForwardScalar)
        return ForwardScalar(
            org.diffkt.digamma(x.primal),
            x.derivativeID,
            org.diffkt.polygamma(1, x.primal) * x.tangent
        )
    }

    override fun polygamma(n: Int, x: DTensor): DTensor {
        require(x is ForwardScalar)
        return ForwardScalar(
            org.diffkt.polygamma(n, x.primal),
            x.derivativeID,
            org.diffkt.polygamma(n + 1, x.primal) * x.tangent
        )
    }

    override fun sqrt(x: DTensor): DTensor {
        require(x is ForwardScalar)
        val primal = org.diffkt.sqrt(x.primal)
        val tangent = x.tangent / (2F * primal)
        return ForwardScalar(primal, x.derivativeID, tangent)
    }

    override fun tanh(x: DTensor): DTensor {
        require(x is ForwardScalar)
        val primal = org.diffkt.tanh(x.primal)
        val tangent = (1F - primal * primal) * x.tangent
        return ForwardScalar(primal, x.derivativeID, tangent)
    }

    override fun meld(values: List<DTensor>, derivativeId: DerivativeID): DTensor {
        return ForwardTensorOperations.meld(values, derivativeId)
    }

    override fun split(x: DTensor, shapes: List<Shape>): List<DTensor> {
        return ForwardTensorOperations.split(x, shapes)
    }

    @AllowUnreduced
    override fun concat(left: DTensor, right: DTensor, axis: Int, derivativeId: DerivativeID): DTensor {
        TODO("DScalar lacks axis $axis")
    }

    @AllowUnreduced
    override fun concat(slices: List<DTensor>, axis: Int, derivativeId: DerivativeID): DTensor {
        TODO("DScalar lacks axis $axis")
    }

    override fun broadcastTo(x: DTensor, newShape: Shape): DTensor {
        return ForwardTensorOperations.broadcastTo(x, newShape)
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
        return ForwardTensorOperations.pow(base, exponent)
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
        require(x is ForwardScalar)
        return ForwardScalar(x.primal.reshape(newShape) as DScalar, x.derivativeID, x.tangent.reshape(x.derivativeID.inputTangentShapeForJacobian))
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
        require(x is ForwardScalar)
        val newPrimal = x.primal.operations.relu(x.primal)
        val tangent = reluGrad(x.primal, x.tangent)
        return ForwardTensor(newPrimal, x.derivativeID, tangent)
    }

    override fun reluGrad(x: DTensor, reluUpstream: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ForwardDerivativeID)
        val up = wrap(reluUpstream, derivativeId)
        val primal = reluGrad(x.primal, up.primal)
        val tangent = reluGrad(x.primal, up.tangent)
        return ForwardTensor(primal, up.derivativeID, tangent)
    }

    override fun sigmoid(x: DTensor): DTensor {
        require(x is ForwardScalar)
        val primal = org.diffkt.sigmoid(x.primal)
        val derivative = primal * (1f - primal).expandToTangent(x.tangent)
        return ForwardScalar(primal, x.derivativeID, x.tangent * derivative)
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
        throw NotImplementedError("Generating a gamma distribution is not differentiable")
    }

    override fun compare(left: DTensor, right: DTensor, comparison: ComparisonKind): DTensor {
        throw IllegalStateException("We should not get here.")
    }

    override fun ifThenElse(condition: DTensor, whenTrue: DTensor, whenFalse: DTensor, derivativeId: DerivativeID): DTensor {
        require(condition is TracingScalar)
        require(derivativeId is ForwardDerivativeID)
        val l = wrap(whenTrue, derivativeId)
        val r = wrap(whenFalse, derivativeId)
        return ForwardScalar(
            ifThenElse(condition, l.primal, r.primal),
            derivativeId,
            ifThenElse(condition, l.tangent, r.tangent))
    }
}
