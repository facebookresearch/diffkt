/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.gpu

import org.diffkt.*
import org.diffkt.StridedFloatTensorOperations
import org.diffkt.external.Gpu
import org.diffkt.random.RandomKey
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

@AllowUnreduced
internal object GpuFloatTensorOperations: Operations {
    override val name get() = "GpuFloatTensor"

    @SType("S: Shape")
    private fun wrap(value: @SType("S") DTensor): @SType("S") GpuFloatTensor {
        if (value is GpuFloatTensor) return value
        TODO("Cannot (automatically) convert to GpuFloatTensor")
    }

    /*
     * Note: In many of these ops, we instantiate GpuFloatTensors with handles
     * returned by the op so the handles will be cleaned up by the GpuFloatTensor
     * finalize() during GC. A Cleaner way to handle this particular case would be
     * to add a Pytorch C++ GPU op hookup that does add with no grad so
     * no grad handles are returned. https://github.com/facebookresearch/diffkt/issues/321
     */
    @SType("S: Shape")
    override fun plus(
            left: @SType("S") DTensor,
            right: @SType("S") DTensor,
            derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        val detachedAndResHandles = Gpu.add(l.handle, r.handle)
        // We explicitly wrap in a GpuFloatTensor for its side-effects.
        /* val detachedLhs = */ GpuFloatTensor(detachedAndResHandles[0])
        /* val detachedRhs = */ GpuFloatTensor(detachedAndResHandles[1])
        val res = GpuFloatTensor(detachedAndResHandles[2])
        return res
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
        val resHandle = Gpu.sub(l.handle, r.handle)
        val res = GpuFloatTensor(resHandle)
        return res
    }

    @SType("S: Shape")
    override fun times(
            left: @SType("S") DTensor,
            right: @SType("S") DTensor,
            derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(derivativeId == NoDerivativeID)
        val l = wrap(left)
        val r = wrap(right)
        val resHandle = Gpu.times(l.handle, r.handle)
        val res = GpuFloatTensor(resHandle)
        return res
    }

    @SType("S: Shape")
    override fun timesScalar(left: DScalar, right: @SType("S") DTensor, derivativeId: DerivativeID): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun zeroOfSameKind(x: DTensor, shape: @SType("S") Shape): @SType("S") DTensor {
        // TODO: can we do this without transferring data (other than the shape) to the GPU?
        return StridedFloatTensorOperations.zeroOfSameKind(FloatScalar.ZERO, shape).to(Device.GPU)
    }

    @SType("S: Shape")
    override fun identityGradientOfSameKind(x: DTensor, halfShape: @SType("S") Shape): @SType("concat(S,S)") DTensor {
        // TODO: can we do this without transferring data (other than the shape) to the GPU?
        return StridedFloatTensorOperations.identityGradientOfSameKind(FloatScalar.ZERO, halfShape).to(Device.GPU)
    }

    @SType("S: Shape")
    override fun unaryMinus(x: @SType("S") DTensor): @SType("S") DTensor {
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
        require(derivativeId == NoDerivativeID)
        require(x.shape == a + b + c)
        require(y.shape == a + c + d)
        val l = wrap(x)
        val r = wrap(y)
        val detachedAndResHandles = Gpu.matmul(l.handle, r.handle)
        /* val detachedLhs = */ GpuFloatTensor(detachedAndResHandles[0])
        /* val detachedRhs = */ GpuFloatTensor(detachedAndResHandles[1])
        val res = GpuFloatTensor(detachedAndResHandles[2])
        require(res.shape == a + b + d)
        return res
    }

    @SType("S1: Shape, S2: Shape")
    override fun outerProduct(
            x: @SType("S1") DTensor,
            y: @SType("S2") DTensor,
            derivativeId: DerivativeID
    ): @SType("concat(S1, S2)") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun sin(x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun cos(x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun tan(x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun atan(x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun exp(x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun ln(x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun lgamma(x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun digamma(x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun polygamma(n: Int, x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun sqrt(x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun tanh(x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    override fun meld(values: List<DTensor>, derivativeId: DerivativeID): DTensor {
        TODO("Not yet implemented")
    }

    override fun split(x: DTensor, shapes: List<Shape>): List<DTensor> {
        TODO("Not yet implemented")
    }

    @SType("S1: Shape, S2: Shape, A: Dim")
    override fun concat(
            left: @SType("S1")  DTensor,
            right: @SType("S2") DTensor,
            axis: @SType("A") Int,
            derivativeId: DerivativeID
    ): @SType("concatOnAxis(S1, S2, A)") DTensor {
        TODO("Not yet implemented")
    }

    override fun concat(slices: List<DTensor>, axis: Int, derivativeId: DerivativeID): DTensor {
        TODO("Not yet implemented")
    }

    override fun broadcastTo(x: DTensor, newShape: Shape): DTensor {
        require(x is GpuFloatTensor)
        val detachedAndResHandles = Gpu.broadcastTo(x.handle, newShape.dims)
        /* val detachedX = */ GpuFloatTensor(detachedAndResHandles[0])
        return GpuFloatTensor(detachedAndResHandles[1])
    }

    override fun convImpl(
        signal: DTensor,
        filter: DTensor,
        hStride: Int,
        vStride: Int,
        padding: Convolve.Padding2D,
        derivativeId: DerivativeID
    ): DTensor {
        TODO("Not yet implemented")
    }

    override fun expand(x: DTensor, newShape: Shape): DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun flip(x: @SType("S") DTensor, axes: IntArray): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    override fun logSoftmax(x: DTensor, axis: Int): DTensor {
        require(x is GpuFloatTensor)
        val detachedAndResHandles = Gpu.logSoftmax(x.handle, axis)
        /* val detachedX = */ GpuFloatTensor(detachedAndResHandles[0])
        return GpuFloatTensor(detachedAndResHandles[1])
    }

    override fun logSoftmaxGrad(x: DTensor, axis: Int, logSoftmax: DTensor, upstream: DTensor): DTensor {
        return GpuFloatTensor(Gpu.logSoftmaxGrad(wrap(upstream).handle, wrap(x).handle, wrap(logSoftmax).handle))
    }

    @SType("S: Shape")
    override fun pow(base: @SType("S") DTensor, exponent: Float): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    override fun view1(x: DTensor, indices: IntArray): DTensor {
        TODO("Not yet implemented")
    }

    override fun view2(x: DTensor, index: Int, axis: Int): DTensor {
        TODO("Not yet implemented")
    }

    override fun view3(x: DTensor, index: IntRange, axis: Int): DTensor {
        TODO("Not yet implemented")
    }

    override fun reshape(x: DTensor, newShape: Shape): DTensor {
        TODO("Not yet implemented")
    }

    override fun reshapeToScalar(x: DTensor): DScalar {
        TODO("Not yet implemented")
    }

    override fun squeeze(x: DTensor, axis: Int): DTensor {
        TODO("Not yet implemented")
    }

    override fun unsqueeze(x: DTensor, axis: Int): DTensor {
        TODO("Not yet implemented")
    }

    override fun transpose(x: DTensor, axes: IntArray): DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun relu(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is GpuFloatTensor)
        val detachedAndResHandles = Gpu.relu(x.handle)
        /* val detachedX = */ GpuFloatTensor(detachedAndResHandles[0])
        return GpuFloatTensor(detachedAndResHandles[1])
    }

    override fun reluGrad(x: DTensor, reluUpstream: DTensor, derivativeId: DerivativeID): DTensor {
        val xx = wrap(x)
        val up = wrap(reluUpstream)
        // This is not very efficient, as we are likely calling relu, which was done
        // previously.  However, this is forced on us by the structure of the underlying API,
        // which was designed specifically to support reverse mode.  A mode efficient
        // implementation is embedded in ReverseTensorOverGpuOperations.relu.
        val relu = relu(xx) as GpuFloatTensor
        return GpuFloatTensor(Gpu.reluGrad(up.handle, xx.handle, relu.handle))
    }

    @SType("S: Shape")
    override fun sigmoid(x: @SType("S") DTensor): @SType("S") DTensor {
        TODO("Not yet implemented")
    }

    override fun sum(x: DTensor, axes: IntArray, keepDims: Boolean): DTensor {
        require(x is GpuFloatTensor)
        val detachedAndResHandles = Gpu.sum(x.handle, axes, keepDims)
        /* val detachedX = */ GpuFloatTensor(detachedAndResHandles[0])
        val resShape = Gpu.getShape(detachedAndResHandles[1])
        return if (resShape.contentEquals(intArrayOf())) GpuFloatScalar(detachedAndResHandles[1]) else GpuFloatTensor(detachedAndResHandles[1])
    }

    override fun avgPool(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        TODO("Not yet implemented")
    }

    override fun avgPoolGrad(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        TODO("Not yet implemented")
    }

    override fun maxPoolWithIndices(
        x: DTensor,
        poolHeight: Int,
        poolWidth: Int,
        withIndices: Boolean
    ): Pair<DTensor, List<IntArray>?> {
        TODO("Not yet implemented")
    }

    override fun gather(x: DTensor, indices: List<Int>, axis: Int, paddingIndex: Int): DTensor {
        TODO("Not yet implemented")
    }

    override fun gatherAtIndices(x: DTensor, indices: List<IntArray>): DTensor {
        TODO("Not yet implemented")
    }

    override fun scatter(x: DTensor, indices: List<Int>, axis: Int, newShape: Shape, paddingIndex: Int): DTensor {
        TODO("Not yet implemented")
    }

    override fun scatterAtIndices(x: DTensor, indices: List<IntArray>, newShape: Shape): DTensor {
        TODO("Not yet implemented")
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
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun ifThenElse(
            condition: @SType("S") DTensor,
            whenTrue: @SType("S") DTensor,
            whenFalse: @SType("S") DTensor,
            derivativeId: DerivativeID
    ): @SType("S") DTensor {
        TODO("Not yet implemented")
    }
}
