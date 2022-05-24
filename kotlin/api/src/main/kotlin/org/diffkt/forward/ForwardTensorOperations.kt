/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.forward

import org.diffkt.*
import org.diffkt.random.RandomKey
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

@AllowUnreduced
internal object ForwardTensorOperations: Operations {
    override val name get() = "ForwardTensor"

    @SType("S: Shape")
    private fun wrap(value: @SType("S") DTensor, derivativeId: ForwardDerivativeID): @SType("S") ForwardTensor {
        if (value is ForwardTensor && value.derivativeID == derivativeId)
            return value
        require(value.derivativeID.sequence < derivativeId.sequence)
        // TODO: the zeroes, below, should be of an appropriate species of tensor, not necessarily a float tensor
        return ForwardTensor(value, derivativeId, FloatTensor.zeros(value.shape + derivativeId.inputTangentShapeForJacobian))
    }

    @SType("S: Shape")
    override fun plus(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") ForwardTensor {
        require(derivativeId is ForwardDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return ForwardTensor(l.primal + r.primal, derivativeId, l.tangent + r.tangent)
    }

    @SType("S: Shape")
    override fun minus(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") ForwardTensor {
        require(derivativeId is ForwardDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return ForwardTensor(l.primal - r.primal, derivativeId, l.tangent - r.tangent)
    }

    @SType("S: Shape")
    override fun times(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") ForwardTensor {
        require(derivativeId is ForwardDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        val newPrimal = l.primal * r.primal
        val newTangent1 = l.primal.expandAndBroadcastToTangent(r.tangent) * r.tangent
        val newTangent2 = r.primal.expandAndBroadcastToTangent(l.tangent) * l.tangent
        return ForwardTensor(newPrimal, derivativeId, newTangent1 + newTangent2)
    }

    @SType("S: Shape")
    override fun timesScalar(left: DScalar, right: @SType("S") DTensor, derivativeId: DerivativeID): @SType("S") DTensor {
        require(derivativeId is ForwardDerivativeID)
        val r = wrap(right, derivativeId)
        val l = ForwardScalarOperations.wrap(left, derivativeId)
        val newPrimal = l.primal * r.primal
        // The tangent we need is of shape A,D where A is the shape of
        // the left tensor and D is the shape of the tangent of a scalar.
        val newTangent1 = l.primal * r.tangent
        val newTangent2 = r.primal outerProduct l.tangent
        return ForwardTensor(newPrimal, derivativeId, newTangent1 + newTangent2)
    }

    @SType("S: Shape")
    override fun zeroOfSameKind(x: DTensor, shape: @SType("S") Shape): @SType("S") ForwardTensor {
        require(x is ForwardTensor)
        val primal = x.primal.operations.zeroOfSameKind(x.primal, shape)
        val tangent = x.tangent.operations.zeroOfSameKind(x.tangent, shape + x.derivativeID.inputTangentShapeForJacobian)
        return ForwardTensor(primal, x.derivativeID, tangent)
    }

    @SType("S: Shape")
    override fun identityGradientOfSameKind(x: DTensor, halfShape: @SType("S") Shape): @SType("concat(S,S)") DTensor {
        return x.primal.operations.identityGradientOfSameKind(x.primal, halfShape)
    }

    @SType("S: Shape")
    override fun unaryMinus(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        return ForwardTensor(-x.primal, x.derivativeID, -x.tangent)
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
        require(derivativeId is ForwardDerivativeID)
        val left = wrap(x, derivativeId)
        val right = wrap(y, derivativeId)
        val newPrimal = left.primal.matmul(right.primal, a, b, c, d)
        val q = left.derivativeID.inputTangentShapeForJacobian
        // left.primal is of Shape(A,B,C)
        // left.tangent is of Shape(A,B,C,Q)
        // right.primal is of Shape(A,C,D)
        // right.tangent is of Shape(A,C,D,Q)
        //
        // newTangent should be of Shape(A,B,D,Q)
        //
        // 1. Combine left (A,B,C) and right.tangent (A,C,D,Q) to make newTangent1 (A,B,D,Q)
        val newTangent1 = left.primal.matmul(right.tangent, a, b, c, d + q) // (A,B,D,Q)
        // 2. Combine left.tangent (A,B,C,Q) and right (A,C,D) to make newTangent2 (A,B,D,Q)
        val t1 = left.tangent.rightTranspose(c, q) // (A,B,Q,C)
        val t2 = t1.matmul(right.primal, a, b + q, c, d) // (A,B,Q,D)
        val newTangent2 = t2.rightTranspose(q, d) // (A,B,D,Q)
        val newTangent = newTangent1 + newTangent2
        return ForwardTensor(newPrimal, left.derivativeID, newTangent)
    }

    @SType("S1: Shape, S2: Shape")
    override fun outerProduct(
        x: @SType("S1") DTensor,
        y: @SType("S2") DTensor,
        derivativeId: DerivativeID
    ): @SType("concat(S1, S2)") DTensor {
        require(derivativeId is ForwardDerivativeID)
        val left = wrap(x, derivativeId)
        val right = wrap(y, derivativeId)
        // left is of shape T<A>; left.tangent is of shape T<A,D>
        // right is of shape T<B>; right.tangent is of shape T<B,D>
        // result primal is of shape T<A,B>
        // result tangent is of shape T<A,B,D>
        // That tangent comes in two parts: left * right.tangent and right * left.tangent.
        val newTangent1 = left.primal outerProduct right.tangent // of shape T<A,B,D>
        val t1 = right.primal outerProduct left.tangent // of shape T<B,A,D>
        val newTangent2 = t1.leftTranspose(right.shape, left.shape) // of shape T<A,B,D>
        return ForwardTensor(left.primal outerProduct right.primal, left.derivativeID, newTangent1 + newTangent2)
    }

    @SType("S: Shape")
    override fun sin(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        return ForwardTensor(org.diffkt.sin(x.primal), x.derivativeID, x.tangent * org.diffkt.cos(x.primal).expandToTangent(x.tangent))
    }

    @SType("S: Shape")
    override fun cos(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        return ForwardTensor(org.diffkt.cos(x.primal), x.derivativeID, x.tangent * org.diffkt.sin(-x.primal).expandToTangent(x.tangent))
    }

    @SType("S: Shape")
    override fun tan(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        val primal = org.diffkt.tan(x.primal)
        val cos = org.diffkt.cos(x.primal)
        val tangent = x.tangent / (cos * cos).expandToTangent(x.tangent)
        return ForwardTensor(primal, x.derivativeID, tangent)
    }

    @SType("S: Shape")
    override fun atan(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        val primal = org.diffkt.atan(x.primal)
        val tangent = x.tangent / (1f + x.primal.pow(2)).expandToTangent(x.tangent)
        return ForwardTensor(primal, x.derivativeID, tangent)
    }

    @SType("S: Shape")
    override fun exp(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        return org.diffkt.exp(x.primal).let { ForwardTensor(it, x.derivativeID, x.tangent * it.expandToTangent(x.tangent)) }
    }

    @SType("S: Shape")
    override fun ln(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        return ForwardTensor(org.diffkt.ln(x.primal), x.derivativeID, x.primal.pow(-1f).expandToTangent(x.tangent) * x.tangent)
    }

    @SType("S: Shape")
    override fun lgamma(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        return ForwardTensor(org.diffkt.lgamma(x.primal), x.derivativeID, org.diffkt.digamma(x.primal).expandToTangent(x.tangent) * x.tangent)
    }

    @SType("S: Shape")
    override fun digamma(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        return ForwardTensor(org.diffkt.digamma(x.primal), x.derivativeID, org.diffkt.polygamma(1, x.primal).expandToTangent(x.tangent) * x.tangent)
    }

    @SType("S: Shape")
    override fun polygamma(n: Int, x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        return ForwardTensor(org.diffkt.polygamma(n, x.primal), x.derivativeID, org.diffkt.polygamma(n + 1, x.primal).expandToTangent(x.tangent) * x.tangent)
    }

    @SType("S: Shape")
    override fun sqrt(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        val primal = org.diffkt.sqrt(x.primal)
        val tangent = x.tangent / (2F * primal).expandToTangent(x.tangent)
        return ForwardTensor(primal, x.derivativeID, tangent)
    }

    @SType("S: Shape")
    override fun tanh(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        val primal = org.diffkt.tanh(x.primal)
        val tangent = (1F - primal * primal).expandToTangent(x.tangent) * x.tangent
        return ForwardTensor(primal, x.derivativeID, tangent)
    }

    override fun meld(values: List<DTensor>, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ForwardDerivativeID)
        val newPrimal = meld(values.map {
            if (it.derivativeID == derivativeId) it.primal else it
        })

        val newGradient = meld(values.map { value: DTensor ->
            if (value.derivativeID == derivativeId) {
                (value as ForwardTensor).tangent
            } else {
                val primalShape = value.shape
                val gradientShape = primalShape + derivativeId.inputTangentShapeForJacobian
                org.diffkt.zeroOfSameKind(value, gradientShape)
            }
        }).reshape(newPrimal.shape + derivativeId.inputTangentShapeForJacobian)
        return ForwardTensor(newPrimal, derivativeId, newGradient)
    }

    override fun split(x: DTensor, shapes: List<Shape>): List<DTensor> {
        require(x is ForwardTensor)
        val primals = x.primal.split(shapes)
        val tangents = x.tangent.split(shapes.map { it + x.derivativeID.inputTangentShapeForJacobian })
        return List(shapes.size) {
            val primal = primals[it]
            val tangent = tangents[it]
            ForwardTensor(primal, x.derivativeID, tangent)
        }
    }

    @SType("S1: Shape, S2: Shape, A: Dim")
    override fun concat(
        left: @SType("S1") DTensor,
        right: @SType("S2") DTensor,
        axis: @SType("A") Int,
        derivativeId: DerivativeID
    ): @SType("concatOnAxis(S1, S2, A)") DTensor {
        require(derivativeId is ForwardDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        val newPrimal = l.primal.concat(r.primal, axis)
        val newTangent = l.tangent.concat(r.tangent, axis)
        return ForwardTensor(newPrimal, derivativeId, newTangent)
    }

    override fun concat(slices: List<DTensor>, axis: Int, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ForwardDerivativeID)
        throw NotImplementedError("Forward derivative of concat on a tensor list is not yet supported")
    }

    override fun broadcastTo(x: DTensor, newShape: Shape): DTensor {
        require(x is ForwardTensor)
        val newPrimal = x.primal.broadcastTo(newShape)
        val newGradient = x.tangent.broadcastTo(newShape + x.derivativeID.inputTangentShapeForJacobian)
        return ForwardTensor(newPrimal, x.derivativeID, newGradient)
    }

    override fun convImpl(
        signal: DTensor,
        filter: DTensor,
        hStride: Int,
        vStride: Int,
        padding: Convolve.Padding2D,
        derivativeId: DerivativeID
    ): DTensor {
        TODO("Convolution does not yet support ForwardTensors")
    }

    override fun expand(x: DTensor, newShape: Shape): DTensor {
        require(x is ForwardTensor)
        val newPrimal = x.primal.expand(newShape)
        val newTangent = x.tangent.expand(newShape + x.derivativeID.inputTangentShapeForJacobian)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    @SType("S: Shape")
    override fun flip(x: @SType("S") DTensor, axes: IntArray): @SType("S") DTensor {
        require(x is ForwardTensor)
        val newPrimal = x.primal.flip(axes)
        val newTangent = x.tangent.flip(axes)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    override fun logSoftmaxGrad(x: DTensor, axis: Int, logSoftmax: DTensor, upstream: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun pow(base: @SType("S") DTensor, exponent: Float): @SType("S") DTensor {
        require(base is ForwardTensor)
        return ForwardTensor(base.primal.pow(exponent), base.derivativeID, exponent * base.primal.pow(exponent - 1).expandToTangent(base.tangent) * base.tangent)
    }

    override fun view1(x: DTensor, indices: IntArray): DTensor {
        require(x is ForwardTensor)
        return ForwardTensor(x.primal.view(indices), x.derivativeID, x.tangent.view(indices))
    }

    override fun view2(x: DTensor, index: Int, axis: Int): DTensor {
        require(x is ForwardTensor)
        return ForwardTensor(x.primal.view(index, axis), x.derivativeID, x.tangent.view(index, axis))
    }

    override fun view3(x: DTensor, index: IntRange, axis: Int): DTensor {
        require(x is ForwardTensor)
        return ForwardTensor(x.primal.view(index, axis), x.derivativeID, x.tangent.view(index, axis))
    }

    override fun reshape(x: DTensor, newShape: Shape): DTensor {
        require(x is ForwardTensor)
        // For DualTensor, we make a recursive reshape call for both the primal
        // and the gradient. The primal is reshaped to the new shape. The
        // gradient is reshaped to `newShape concat functionInputShape`, where
        // functionInputShape is the shape of the input of the function
        // we are taking the derivative of.
        //
        // Note that if this is the last operation in the function we are
        // taking the derivative of, the shape will be `functionOutputShape
        // concat functionInputShape`, so we will have the derivative of each
        // of the function's outputs with respect to each of the function's
        // inputs.
        // Concrete examples of this can be found in FlattenTest.
        assert(x.primal.shape.isPrefix(x.tangent.shape))
        val newPrimal = x.primal.reshape(newShape)
        val newTangentShape = newShape + x.derivativeID.inputTangentShapeForJacobian
        val newTangent = x.tangent.reshape(newTangentShape)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    override fun reshapeToScalar(x: DTensor): DScalar {
        require(x is ForwardTensor)
        val primal = x.primal.operations.reshapeToScalar(x.primal)
        val tangent = x.tangent.reshape(x.derivativeID.inputTangentShapeForJacobian)
        return ForwardScalar(primal, x.derivativeID, tangent)
    }

    override fun squeeze(x: DTensor, axis: Int): DTensor {
        require(x is ForwardTensor)
        val newPrimal = x.primal.squeeze(axis)
        val newTangent = x.tangent.squeeze(axis)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    override fun unsqueeze(x: DTensor, axis: Int): DTensor {
        require(x is ForwardTensor)
        val newPrimal = x.primal.unsqueeze(axis)
        val newTangent = x.tangent.unsqueeze(axis)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    override fun transpose(x: DTensor, axes: IntArray): DTensor {
        require(x is ForwardTensor)
        fun extendAxes(axes: IntArray, additional: Int): IntArray =
            IntArray(axes.size + additional) { i -> if (i < axes.size) axes[i] else i }
        return ForwardTensor(x.primal.transpose(axes), x.derivativeID, x.tangent.transpose(extendAxes(axes, x.derivativeID.inputTangentShapeForJacobian.rank)))
    }

    @SType("S: Shape")
    override fun relu(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        val newPrimal = x.primal.operations.relu(x.primal)
        val tangent = reluGrad(x.primal, x.tangent)
        return ForwardTensor(newPrimal, x.derivativeID, tangent)
    }

    override fun reluGrad(x: DTensor, reluUpstream: DTensor, derivativeId: DerivativeID): DTensor {
        require(x is ForwardTensor)
        val up = wrap(reluUpstream, x.derivativeID)
        // TODO: is x.tangent needed for anything?
        val primal = reluGrad(x.primal, up.primal)
        val tangent = reluGrad(x.primal, up.tangent)
        return ForwardTensor(primal, up.derivativeID, tangent)
    }

    @SType("S: Shape")
    override fun sigmoid(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ForwardTensor)
        val result = org.diffkt.sigmoid(x.primal)
        val derivative = (result * (1f - result)).expandToTangent(x.tangent)
        return ForwardTensor(result, x.derivativeID, x.tangent * derivative)
    }

    override fun sum(x: DTensor, axes: IntArray, keepDims: Boolean): DTensor {
        require(x is ForwardTensor)
        val newPrimal = x.primal.sum(axes, keepDims)
        val newTangent = x.tangent.sum(axes, keepDims)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    override fun avgPool(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        require(x is ForwardTensor)
        val newPrimal = org.diffkt.model.avgPool(x.primal, poolHeight, poolWidth)
        val newTangent = org.diffkt.model.avgPool(x.tangent, poolHeight, poolWidth)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    override fun avgPoolGrad(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        require(x is ForwardTensor)
        val newPrimal = org.diffkt.model.avgPoolGrad(x.primal, poolHeight, poolWidth)
        val newTangent = org.diffkt.model.avgPoolGrad(x.tangent, poolHeight, poolWidth)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    override fun maxPoolWithIndices(
        x: DTensor,
        poolHeight: Int,
        poolWidth: Int,
        withIndices: Boolean
    ): Pair<DTensor, List<IntArray>?> {
        require(x is ForwardTensor)
        val (newPrimal, indices) = x.primal.operations.maxPoolWithIndices(
            x.primal,
            poolHeight,
            poolWidth,
            withIndices = true
        )
        val newTangent = x.tangent.operations.gatherAtIndices(x.tangent, indices!!).reshape(newPrimal.shape + x.derivativeID.inputTangentShapeForJacobian)
        val result = ForwardTensor(newPrimal, x.derivativeID, newTangent)
        return Pair(result, indices)
    }

    override fun gather(x: DTensor, indices: List<Int>, axis: Int, paddingIndex: Int): DTensor {
        require(x is ForwardTensor)
        val newPrimal = x.primal.operations.gather(x.primal, indices, axis, paddingIndex)
        val newTangent = x.tangent.operations.gather(x.tangent, indices, axis, paddingIndex)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    override fun gatherAtIndices(x: DTensor, indices: List<IntArray>): DTensor {
        require(x is ForwardTensor)
        val newPrimal = x.primal.operations.gatherAtIndices(x.primal, indices)
        val newTangent = x.tangent.operations.gatherAtIndices(x.tangent, indices)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    override fun scatter(x: DTensor, indices: List<Int>, axis: Int, newShape: Shape, paddingIndex: Int): DTensor {
        require(x is ForwardTensor)
        val newPrimal = x.primal.operations.scatter(x.primal, indices, axis, newShape, paddingIndex)
        val newTangent = x.tangent.operations.scatter(x.tangent, indices, axis, newShape + x.derivativeID.inputTangentShapeForJacobian, paddingIndex)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    override fun scatterAtIndices(x: DTensor, indices: List<IntArray>, newShape: Shape): DTensor {
        require(x is ForwardTensor)
        val newPrimal = x.primal.operations.scatterAtIndices(x.primal, indices, newShape)
        val newTangent = x.tangent.operations.scatterAtIndices(x.tangent, indices, newShape + x.derivativeID.inputTangentShapeForJacobian)
        return ForwardTensor(newPrimal, x.derivativeID, newTangent)
    }

    override fun gamma(alpha: DTensor, randomKey: RandomKey): DTensor {
        throw NotImplementedError("Generating a gamma distribution is not differentiable")
    }

    @SType("S: Shape")
    override fun compare(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        comparison: ComparisonKind
    ): @SType("S") DTensor {
        throw IllegalStateException("We should not get here.")
    }

    @SType("S: Shape")
    override fun ifThenElse(
        condition: @SType("S") DTensor,
        whenTrue: @SType("S") DTensor,
        whenFalse: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(derivativeId is ForwardDerivativeID)
        val l = wrap(whenTrue, derivativeId)
        val r = wrap(whenFalse, derivativeId)
        val leftTangent = l.tangent
        val conditionForTangent = if (condition.isScalar) condition else condition.expandAndBroadcastToTangent(leftTangent)
        return ForwardTensor(
            ifThenElse(condition, l.primal, r.primal),
            derivativeId,
            ifThenElse(conditionForTangent, leftTangent, r.tangent)
        )
    }
}
