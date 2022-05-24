/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import org.diffkt.*
import org.diffkt.model.BatchNormResult
import org.diffkt.random.RandomKey
import shapeTyping.annotations.AllowUnreduced

@AllowUnreduced
internal open class ReverseScalarOperationsImpl: Operations {
    override val name get() = "ReverseScalar"
    override val tensor: Operations get() = ReverseTensorOperations
    internal fun wrap(value: DTensor, derivativeId: ReverseDerivativeID): ReverseScalar {
        require(value is DScalar)
        if (value is ReverseScalar && value.derivativeID == derivativeId)
            return value
        require(value.derivativeID.sequence < derivativeId.sequence)
        return object : ReverseScalar(primal = value, derivativeID = derivativeId) {
            override fun backpropagate() { }
        }
    }

    override fun plus(left: DTensor, right: DTensor, derivativeId: DerivativeID): ReverseScalar {
        require(derivativeId is ReverseDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return object : ReverseScalar(l.primal + r.primal, derivativeId) {
            override fun backpropagate() {
                assert(upstream.derivativeID.sequence < derivativeId.sequence)
                l.pushback(upstream)
                r.pushback(upstream)
            }
        }
    }

    override fun minus(left: DTensor, right: DTensor, derivativeId: DerivativeID): ReverseScalar {
        require(derivativeId is ReverseDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return object : ReverseScalar(l.primal - r.primal, derivativeId) {
            override fun backpropagate() {
                assert(upstream.derivativeID.sequence < derivativeId.sequence)
                l.pushback(upstream)
                r.pushback(-upstream)
            }
        }
    }

    override fun times(left: DTensor, right: DTensor, derivativeId: DerivativeID): DScalar {
        require(derivativeId is ReverseDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return object : ReverseScalar(l.primal * r.primal, derivativeId) {
            override fun backpropagate() {
                assert(upstream.derivativeID.sequence < derivativeId.sequence)
                assert(upstream.shape == l.shape + derivativeID.upstreamShape)
                l.pushback(r.primal * upstream)
                r.pushback(l.primal * upstream)
            }
        }
    }

    override fun timesScalar(left: DScalar, right: DTensor, derivativeId: DerivativeID): DTensor {
        return times(left, right, derivativeId)
    }

    override fun zeroOfSameKind(x: DTensor, shape: Shape): DTensor {
        require(x is ReverseScalar)
        return object : ReverseScalar(x.primal.operations.zeroOfSameKind(x.primal, shape) as DScalar, x.derivativeID) {
            override fun backpropagate() {
            }
        }
    }

    override fun identityGradientOfSameKind(x: DTensor, halfShape: Shape): DTensor {
        return x.primal.operations.identityGradientOfSameKind(x.primal, halfShape)
    }

    override fun unaryMinus(x: DTensor): ReverseScalar {
        require(x is ReverseScalar)
        return object : ReverseScalar(-x.primal, x.derivativeID) {
            override fun backpropagate() {
                assert(upstream.derivativeID.sequence < x.derivativeID.sequence)
                x.pushback(-upstream)
            }
        }
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
        require(x is ReverseScalar)
        return object : ReverseScalar(org.diffkt.sin(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * org.diffkt.cos(x.primal))
            }
        }
    }

    override fun cos(x: DTensor): DTensor {
        require(x is ReverseScalar)
        return object : ReverseScalar(org.diffkt.cos(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * org.diffkt.sin(-x.primal))
            }
        }
    }

    override fun tan(x: DTensor): DTensor {
        require(x is ReverseScalar)
        return object : ReverseScalar(org.diffkt.tan(x.primal), x.derivativeID) {
            override fun backpropagate() {
                val cos = org.diffkt.cos(x.primal)
                x.pushback(upstream / (cos * cos))
            }
        }
    }

    override fun atan(x: DTensor): DTensor {
        require(x is ReverseScalar)
        return object : ReverseScalar(org.diffkt.atan(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream / (1f + x.primal.pow(2)))
            }
        }
    }

    override fun exp(x: DTensor): DTensor {
        require(x is ReverseScalar)
        return object : ReverseScalar(org.diffkt.exp(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * this.primal)
            }
        }
    }

    override fun ln(x: DTensor): DTensor {
        require(x is ReverseScalar)
        return object : ReverseScalar(org.diffkt.ln(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream / x.primal)
            }
        }
    }

    override fun lgamma(x: DTensor): DTensor {
        require(x is ReverseScalar)
        return object : ReverseScalar(org.diffkt.lgamma(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * org.diffkt.digamma(x.primal))
            }
        }
    }

    override fun digamma(x: DTensor): DTensor {
        require(x is ReverseScalar)
        return object : ReverseScalar(org.diffkt.digamma(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * org.diffkt.polygamma(1, x.primal))
            }
        }
    }

    override fun polygamma(n: Int, x: DTensor): DTensor {
        require(x is ReverseScalar)
        return object : ReverseScalar(org.diffkt.polygamma(n, x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * org.diffkt.polygamma(n + 1, x.primal))
            }
        }
    }

    override fun sqrt(x: DTensor): DTensor {
        require(x is ReverseScalar)
        val primal = org.diffkt.sqrt(x.primal)
        return object : ReverseScalar(primal, x.derivativeID) {
            override fun backpropagate() {
                val tangent = upstream / (2F * primal)
                x.pushback(tangent)
            }
        }
    }

    override fun tanh(x: DTensor): DTensor {
        require(x is ReverseScalar)
        val primal = org.diffkt.tanh(x.primal)
        return object : ReverseScalar(primal, x.derivativeID) {
            override fun backpropagate() {
                val tangent = (1F - primal * primal) * upstream
                x.pushback(tangent)
            }
        }
    }

    override fun meld(values: List<DTensor>, derivativeId: DerivativeID): DTensor {
        return ReverseTensorOperations.meld(values, derivativeId)
    }

    override fun split(x: DTensor, shapes: List<Shape>): List<DTensor> {
        return ReverseTensorOperations.split(x, shapes)
    }

    override fun concat(left: DTensor, right: DTensor, axis: Int, derivativeId: DerivativeID): DTensor {
        TODO("DScalar lacks axis $axis")
    }

    override fun concat(slices: List<DTensor>, axis: Int, derivativeId: DerivativeID): DTensor {
        TODO("DScalar lacks axis $axis")
    }

    override fun broadcastTo(x: DTensor, newShape: Shape): DTensor {
        return ReverseTensorOperations.broadcastTo(x, newShape)
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
        throw IllegalArgumentException("Cannot run logSoftmax on a scalar.")
    }

    override fun pow(base: DTensor, exponent: Float): DTensor {
        require(base is ReverseScalar)
        return object : ReverseScalar(base.primal.pow(exponent), base.derivativeID) {
            override fun backpropagate() {
                assert(upstream.shape == base.shape + derivativeID.upstreamShape)
                assert(upstream.derivativeID.sequence < derivativeID.sequence)
                base.pushback(exponent * base.primal.pow(exponent - 1).expandToTangent(upstream) * upstream)
            }
        }
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
        return ReverseTensorOperations.reshape(x, newShape)
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
        require(x is ReverseScalar)
        return object : ReverseScalar(x.primal.relu(), x.derivativeID) {
            override fun backpropagate() {
                val grad = reluGrad(x.primal, upstream)
                x.pushback(grad)
            }
        }
    }

    override fun reluGrad(x: DTensor, reluUpstream: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ReverseDerivativeID)
        val up = wrap(reluUpstream, derivativeId)
        return object : ReverseScalar(reluGrad(x.primal, up.primal) as DScalar, derivativeId) {
            override fun backpropagate() {
                // TODO: should there be a call to x.pushback(...) ?
                val upd = reluGrad(x.primal, upstream)
                up.pushback(upd)
            }
        }
    }

    override fun sigmoid(x: DTensor): DTensor {
        require(x is ReverseScalar)
        return object : ReverseScalar(org.diffkt.sigmoid(x.primal), x.derivativeID) {
            override fun backpropagate() {
                val derivative = primal * (1f - primal).expandToTangent(upstream)
                x.pushback(upstream * derivative)
            }
        }
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

    override fun div(left: DTensor, right: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ReverseDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return object : ReverseScalar(l.primal/r.primal, derivativeId) {
            override fun backpropagate() {
                l.pushback(upstream / r.primal)
                r.pushback(-upstream * this.primal/r.primal)
            }
        }
    }

    override fun gamma(alpha: DTensor, randomKey: RandomKey): DTensor {
        throw NotImplementedError("Generating a gamma distribution is not differentiable")
    }

    override fun compare(left: DTensor, right: DTensor, comparison: ComparisonKind): DTensor {
        throw IllegalStateException("Should not happen; caller should have reduced to primal")
    }

    override fun ifThenElse(condition: DTensor, whenTrue: DTensor, whenFalse: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ReverseDerivativeID)
        require(condition is DScalar)
        val l = wrap(whenTrue, derivativeId)
        val r = wrap(whenFalse, derivativeId)
        val primal = ifThenElse(condition, l.primal, r.primal)
        return object : ReverseScalar(primal, derivativeId) {
            override fun backpropagate() {
                if (condition is FloatScalar) {
                    if (condition.value > 0f) {
                        l.pushback(upstream)
                    } else {
                        r.pushback(upstream)
                    }
                } else {
                    val zeros = FloatTensor.zeros(upstream.shape)
                    l.pushback(ifThenElse(condition, upstream, zeros))
                    r.pushback(ifThenElse(condition, zeros, upstream))
                }
            }
        }
    }
}
