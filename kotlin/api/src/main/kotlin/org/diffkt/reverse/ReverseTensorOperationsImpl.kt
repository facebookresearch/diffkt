/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import org.diffkt.*
import org.diffkt.Broadcasting
import org.diffkt.convImpl
import org.diffkt.external.Dnnl
import org.diffkt.isDnnlEligible
import org.diffkt.model.BatchNormResult
import org.diffkt.model.baseBatchNorm
import org.diffkt.model.maxPoolWithIndicesDnnl
import org.diffkt.random.RandomKey
import org.diffkt.shouldSendToCpp
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

@AllowUnreduced
internal open class ReverseTensorOperationsImpl: Operations {
    override val name get() = "ReverseTensor"

    @SType("S: Shape")
    private fun wrap(value: @SType("S") DTensor, derivativeId: ReverseDerivativeID): @SType("S") ReverseTensor {
        if (value is ReverseTensor && value.derivativeID == derivativeId)
            return value
        require(value.derivativeID.sequence < derivativeId.sequence)
        require(value !is DScalar)
        return object : ReverseTensor(primal = value, derivativeID = derivativeId) {
            override fun backpropagate() { }
        }
    }

    @SType("S: Shape")
    override fun plus(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") ReverseTensor {
        require(derivativeId is ReverseDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return object : ReverseTensor(l.primal + r.primal, derivativeId) {
            override fun backpropagate() {
                assert(upstream.derivativeID.sequence < derivativeID.sequence)
                l.pushback(upstream)
                r.pushback(upstream)
            }
        }
    }

    @SType("S: Shape")
    override fun minus(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") ReverseTensor {
        require(derivativeId is ReverseDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return object : ReverseTensor(l.primal - r.primal, derivativeId) {
            override fun backpropagate() {
                assert(upstream.derivativeID.sequence < derivativeID.sequence)
                l.pushback(upstream)
                r.pushback(-upstream)
            }
        }
    }

    @SType("S: Shape")
    override fun times(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(derivativeId is ReverseDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return object : ReverseTensor(l.primal * r.primal, derivativeId) {
            override fun backpropagate() {
                assert(upstream.derivativeID.sequence < derivativeID.sequence)
                l.pushback(r.primal.expandToTangent(upstream) * upstream)
                r.pushback(l.primal.expandToTangent(upstream) * upstream)
            }
        }
    }

    @SType("S: Shape")
    override fun timesScalar(
        left: DScalar,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(derivativeId is ReverseDerivativeID)
        val l = ReverseScalarOperations.wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        return object : ReverseTensor(l.primal * r.primal, derivativeId) {
            override fun backpropagate() {
                // right is shape T<R>
                // upstream is shape T<R,D>
                assert(upstream.shape == r.primal.shape + derivativeID.upstreamShape)

                // push a tangent of shape T<D> to the left
                l.pushback(r.primal.innerProduct(r.primal.shape, upstream))

                // push tangent of shape T<R,D> to the right
                r.pushback(l.primal * upstream)
            }
        }
    }

    @SType("S: Shape")
    override fun zeroOfSameKind(x: DTensor, shape: @SType("S") Shape): @SType("S") DTensor {
        require(x is ReverseTensor)
        return x.primal.operations.zeroOfSameKind(x.primal, shape)
    }
    @SType("S: Shape")
    override fun identityGradientOfSameKind(x: DTensor, halfShape: @SType("S") Shape): @SType("concat(S,S)") DTensor {
        return x.primal.operations.identityGradientOfSameKind(x.primal, halfShape)
    }

    @SType("S: Shape")
    override fun unaryMinus(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(-x.primal, x.derivativeID) {
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
        require(derivativeId is ReverseDerivativeID)
        val left = wrap(x, derivativeId)
        val right = wrap(y, derivativeId)
        val newPrimal = left.primal.matmul(right.primal, a, b, c, d)
        return object : ReverseTensor(newPrimal, left.derivativeID) {
            override fun backpropagate() {
                val q = left.derivativeID.upstreamShape
                // upstream is of Shape(A,B,D,Q)
                // left.primal is of Shape(A,B,C) and needs an upstream of Shape(A,B,C,Q)
                // right.primal is of Shape(A,C,D) and needs an upstream of Shape(A,C,D,Q)
                //
                // 1. Combine upstream (A,B,D,Q) and right.primal (A,C,D) to get leftUpstream (A,B,C,Q)
                val t1 = upstream.rightTranspose(d, q) // (A,B,Q,D)
                val t2 = right.primal.rightTranspose(c, d) // (A,D,C)
                val t3 = t1.matmul(t2, a, b + q, d, c) // (A,B,Q,C)
                val leftUpstream = t3.rightTranspose(q, c) // (A,B,C,Q) as required
                left.pushback(leftUpstream)
                // 2. Combine upstream (A,B,D,Q) and left.primal (A,B,C) to get rightUpstream (A,C,D,Q)
                val t4 = left.primal.rightTranspose(b, c) // (A,C,B)
                val rightUpstream = t4.matmul(upstream, a, c, b, d + q) // (A,C,D,Q) as required
                right.pushback(rightUpstream)
            }
        }
    }

    @SType("S1: Shape, S2: Shape")
    override fun outerProduct(
        x: @SType("S1") DTensor,
        y: @SType("S2") DTensor,
        derivativeId: DerivativeID
    ): @SType("concat(S1, S2)") DTensor {
        require(derivativeId is ReverseDerivativeID)
        val left = wrap(x, derivativeId)
        val right = wrap(y, derivativeId)
        val resultPrimal = left.primal outerProduct right.primal
        return object : ReverseTensor(resultPrimal, left.derivativeID) {
            override fun backpropagate() {
                // upstream is of shape T<A,B,D>
                // left is of shape T<A>; it needs a gradient of shape T<A,D>
                // right is of shape T<B>; it needs a gradient of shape T<B,D>
                val t1 = upstream.leftTranspose(left.shape, right.shape) // of shape T<B,A,D>
                val t2 = right.primal.innerProduct(right.shape, t1) // of shape T<A,D>
                left.pushback(t2)
                val t3 = left.primal.innerProduct(left.shape, upstream) // of shape T<A,D>
                right.pushback(t3)
            }
        }
    }

    @SType("S: Shape")
    override fun sin(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(org.diffkt.sin(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * org.diffkt.cos(x.primal).expandToTangent(upstream))
            }
        }
    }

    @SType("S: Shape")
    override fun cos(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(org.diffkt.cos(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * org.diffkt.sin(-x.primal).expandToTangent(upstream))
            }
        }
    }

    @SType("S: Shape")
    override fun tan(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(org.diffkt.tan(x.primal), x.derivativeID) {
            override fun backpropagate() {
                val cos = org.diffkt.cos(x.primal)
                x.pushback(upstream / (cos * cos).expandToTangent(upstream))
            }
        }
    }

    @SType("S: Shape")
    override fun atan(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(org.diffkt.atan(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream / (1f + x.primal.pow(2)).expandToTangent(upstream))
            }
        }
    }

    @SType("S: Shape")
    override fun exp(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(org.diffkt.exp(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * this.primal.expandToTangent(upstream))
            }
        }
    }

    @SType("S: Shape")
    override fun ln(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(org.diffkt.ln(x.primal), x.derivativeID) {
            override fun backpropagate() {
                assert(upstream.shape == x.shape + derivativeID.upstreamShape)
                assert(upstream.derivativeID.sequence < derivativeID.sequence)
                x.pushback(x.primal.pow(-1f).expandToTangent(upstream) * upstream)
            }
        }
    }

    @SType("S: Shape")
    override fun lgamma(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(org.diffkt.lgamma(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * org.diffkt.digamma(x.primal).expandToTangent(upstream))
            }
        }
    }

    @SType("S: Shape")
    override fun digamma(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(org.diffkt.digamma(x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * org.diffkt.polygamma(1, x.primal).expandToTangent(upstream))
            }
        }
    }

    @SType("S: Shape")
    override fun polygamma(n: Int, x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(org.diffkt.polygamma(n, x.primal), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream * org.diffkt.polygamma(n + 1, x.primal).expandToTangent(upstream))
            }
        }
    }

    @SType("S: Shape")
    override fun sqrt(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        val primal = org.diffkt.sqrt(x.primal)
        return object : ReverseTensor(primal, x.derivativeID) {
            override fun backpropagate() {
                val tangent = upstream / (2F * primal).expandToTangent(upstream)
                x.pushback(tangent)
            }
        }
    }

    @SType("S: Shape")
    override fun tanh(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        val primal = org.diffkt.tanh(x.primal)
        return object : ReverseTensor(primal, x.derivativeID) {
            override fun backpropagate() {
                val tangent = (1F - primal * primal).expandToTangent(upstream) * upstream
                x.pushback(tangent)
            }
        }
    }

    override fun meld(values: List<DTensor>, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ReverseDerivativeID)
        val newPrimal = meld(values.map {
            if (it.derivativeID == derivativeId) it.primal else it
        })
        return object : ReverseTensor(newPrimal, derivativeId) {
            override fun backpropagate() {
                // We need to take the upstream and break it up and backpropagate it to the inputs.
                val splitUpstream = upstream.split(values.map { it.shape + derivativeID.upstreamShape })
                for (i in values.indices) {
                    val meldedValue = values[i]
                    if (meldedValue.derivativeID == derivativeID) {
                        meldedValue as ReverseTensor
                        meldedValue.pushback(splitUpstream[i])
                    }
                }
            }
        }
    }

    override fun split(x: DTensor, shapes: List<Shape>): List<DTensor> {
        require(x is ReverseTensor)
        // this intermediate object exists only to hold the backpropagation values in splitUpstreams
        val splitReverseTensor = object : ReverseTensor(x.primal, x.derivativeID)
        {
            val splitUpstreams = Array<DTensor?>(shapes.size) { null }

            val splitResult = run {
                val primals = this.primal.split(shapes)
                List(shapes.size) {
                    val onePrimal = primals[it]
                    when (onePrimal) {
                        is DScalar -> object : ReverseScalar(onePrimal, derivativeID) {
                            override fun backpropagate() {
                                splitUpstreams[it] = upstream
                            }
                        }
                        else -> object : ReverseTensor(onePrimal, derivativeID) {
                            override fun backpropagate() {
                                splitUpstreams[it] = upstream
                            }
                        }
                    }
                }
            }

            override fun backpropagate() {
                assert(!hasUpstream)
                this.upstream = meld(splitUpstreams.map { it!! })
                    .reshape(x.shape + derivativeID.upstreamShape)
                for (i in splitUpstreams.indices) splitUpstreams[i] = null
                x.pushback(upstream)
            }
        }
        return splitReverseTensor.splitResult
    }

    @SType("S1: Shape, S2: Shape, A: Dim")
    override fun concat(
        left: @SType("S1") DTensor,
        right: @SType("S2") DTensor,
        axis: @SType("A") Int,
        derivativeId: DerivativeID
    ): @SType("concatOnAxis(S1, S2, A)") DTensor {
        require(derivativeId is ReverseDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        val primal = l.primal.concat(r.primal, axis)
        return object : ReverseTensor(primal, derivativeId) {
            override fun backpropagate() {
                val endIx = upstream.shape[axis]
                l.pushback(upstream.slice(0, left.shape[axis], axis = axis))
                r.pushback(upstream.slice(endIx - right.shape[axis], endIx, axis = axis))
            }
        }
    }

    override fun concat(slices: List<DTensor>, axis: Int, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ReverseDerivativeID)
        val primals = slices.map {
            if (it.derivativeID == derivativeId) it.primal else it
        }
        val primal = concat(primals, axis)
        val startIndices = slices.scan(0) { idx, t -> idx + t.shape[axis] }.dropLast(1)
        val inputs = slices.zip(startIndices).filter { pair -> pair.first.derivativeID == derivativeId }
        return object : ReverseTensor(primal, derivativeId) {
            override fun backpropagate() {
                inputs.forEach {
                    val (t, startIdx) = it
                    t as ReverseTensor
                    t.pushback(upstream.slice(startIdx, startIdx + t.shape[axis], axis))
                }
            }
        }
    }

    override fun broadcastTo(x: DTensor, newShape: Shape): DTensor {
        x as ReverseTensor
        return object : ReverseTensor(x.primal.broadcastTo(newShape), x.derivativeID) {
            override fun backpropagate() {
                val axes = Broadcasting.getBroadcastedAxes(x.shape, newShape)
                val rankDiff = newShape.rank - x.rank
                val pushback = upstream.sum(axes, keepDims = true).sum(IntArray(rankDiff) { it })
                x.pushback(pushback)
            }
        }
    }

    /** Helper function for the gradient of signal for conv2D **/
    private fun conv2dSignalDiff(
        signal: StridedFloatTensor,
        filter: StridedFloatTensor,
        seed: StridedFloatTensor,
        hstride: Int,
        vstride: Int,
        inputPadding: Convolve.Padding2D
    ): FloatTensor {
        val normalizedSeed = seed.normalize()
        return StridedFloatTensor.contiguous(signal.shape) {
            Dnnl.conv2dGradImage(
                // result (image grad)
                signal.shape.dims,
                it,
                // seed
                normalizedSeed.shape.dims,
                normalizedSeed.data,
                // filter
                filter.shape.dims,
                filter.data,
                // strides
                vstride, // height
                hstride, // width
                // padding
                inputPadding.left,
                inputPadding.right,
                inputPadding.top,
                inputPadding.bottom
            )

        }
    }
    /** Helper function for the gradient of filter for conv2D **/
    private fun conv2dFilterDiff(
        signal: StridedFloatTensor,
        filter: StridedFloatTensor,
        seed: StridedFloatTensor,
        hstride: Int,
        vstride: Int,
        padding: Convolve.Padding2D
    ): FloatTensor {
        val normalizedSeed = seed.normalize()
        return StridedFloatTensor.contiguous(filter.shape) {
            Dnnl.conv2dGradFilter(
                // result (filter grad)
                filter.shape.dims,
                it,
                // seed
                normalizedSeed.shape.dims,
                normalizedSeed.data,
                // signal
                signal.shape.dims,
                signal.data,
                // strides
                vstride, // height
                hstride, // width
                // padding
                padding.left,
                padding.right,
                padding.top,
                padding.bottom
            )
        }
    }
    override fun convImpl(
        signal: DTensor,
        filter: DTensor,
        hStride: Int,
        vStride: Int,
        padding: Convolve.Padding2D,
        derivativeId: DerivativeID
    ): DTensor {
        require(derivativeId is ReverseDerivativeID)
        val s = wrap(signal, derivativeId)
        val f = wrap(filter, derivativeId)
        return object : ReverseTensor(convImpl(s.primal, f.primal, hStride, vStride, padding), derivativeId) {
            init {
                require (s.primal is FloatTensor && f.primal is FloatTensor)
                { "Higher order conv gradient is not supported" }
            }
            override fun backpropagate() {
                require(derivativeID.upstreamShape == Shape()) { "Convolution gradients are only supported for a function that returns a scalar" }
                assert(upstream.derivativeID.sequence < derivativeID.sequence)
                require(upstream is StridedFloatTensor) { "Higher order conv gradient not supported" }
                val signalPrimal = (s.primal as FloatTensor).normalize()
                val filterPrimal = (f.primal as FloatTensor).normalize()
                val upstreamT = upstream as StridedFloatTensor
                s.pushback(conv2dSignalDiff(signalPrimal, filterPrimal, upstreamT, hStride, vStride, padding))
                f.pushback(conv2dFilterDiff(signalPrimal, filterPrimal, upstreamT, hStride, vStride, padding))
            }
        }
    }

    override fun expand(x: DTensor, newShape: Shape): DTensor {
        require(x is ReverseTensor)
        val newPrimal = x.primal.expand(newShape)
        return object : ReverseTensor(newPrimal, x.derivativeID) {
            /**
             * Helper function to do compute the axes expanded by expand. Used in the reverse gradient calculation of expand.
             *
             * For example,
             * t = Tensor(2, 1, 4).expand([2, 3, 4]) // Tensor(2, 3, 4)
             *
             * The only dim expanded is dim 1 (1 is expanded to 3 at dim one). So this helper would return [1]
             */
            private fun getExpandedAxes(oldShape: Shape, expandedShape: Shape): IntArray {
                val axesExpanded = mutableListOf<Int>()
                for (index in expandedShape.indices) {
                    val newDim = expandedShape[index]
                    if (oldShape[index] != newDim && newDim != -1)
                        axesExpanded += index
                }
                return axesExpanded.toIntArray()
            }
            override fun backpropagate() {
                val axes = getExpandedAxes(x.shape, newShape)
                val g = upstream.sum(axes, keepDims = true)
                x.pushback(g)
            }
        }
    }

    @SType("S: Shape")
    override fun flip(x: @SType("S") DTensor, axes: IntArray): @SType("S") DTensor {
        require(x is ReverseTensor)
        val newPrimal = x.primal.flip(axes)
        return object : ReverseTensor(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream.flip(axes))
            }
        }
    }

    override fun logSoftmax(x: DTensor, axis: Int): DTensor {
        require(x is ReverseTensor)
        val xPrimal = x.primal
        val newPrimal = xPrimal.operations.logSoftmax(xPrimal, axis)
        return object : ReverseTensor(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                val grad = xPrimal.operations.logSoftmaxGrad(xPrimal, axis, newPrimal, upstream)
                x.pushback(grad)
            }
        }
    }

    override fun logSoftmaxGrad(x: DTensor, axis: Int, logSoftmax: DTensor, upstream: DTensor): DTensor {
        TODO("Not yet implemented")
    }

    @SType("S: Shape")
    override fun pow(base: @SType("S") DTensor, exponent: Float): @SType("S") DTensor {
        require(base is ReverseTensor)
        return object : ReverseTensor(base.primal.pow(exponent), base.derivativeID) {
            override fun backpropagate() {
                assert(upstream.shape == base.shape + derivativeID.upstreamShape)
                assert(upstream.derivativeID.sequence < derivativeID.sequence)
                base.pushback(exponent * base.primal.pow(exponent - 1).expandToTangent(upstream) * upstream)
            }
        }
    }

    override fun view1(x: DTensor, indices: IntArray): DTensor {
        require(x is ReverseTensor)
        val newPrimal = x.primal.view(indices)
        return if (newPrimal is DScalar)
            object : ReverseScalar(newPrimal, x.derivativeID) {
                override fun backpropagate() {
                    val originalGradientShape = x.shape + derivativeID.upstreamShape
                    x.pushback(org.diffkt.zeroOfSameKind(newPrimal, originalGradientShape).withChange(indices, upstream))
                }
            }
        else
            object : ReverseTensor(newPrimal, x.derivativeID) {
                override fun backpropagate() {
                    val originalGradientShape = x.shape + derivativeID.upstreamShape
                    x.pushback(org.diffkt.zeroOfSameKind(newPrimal, originalGradientShape).withChange(indices, upstream))
                }
            }
    }

    override fun view2(x: DTensor, index: Int, axis: Int): DTensor {
        require(x is ReverseTensor)
        val newPrimal = x.primal.view(index, axis)
        return if (newPrimal is DScalar)
            object : ReverseScalar(newPrimal, x.derivativeID) {
                override fun backpropagate() {
                    val originalGradientShape = x.shape + derivativeID.upstreamShape
                    x.pushback(
                        org.diffkt.zeroOfSameKind(newPrimal, originalGradientShape).withChange(index, axis, upstream))
                }
            }
        else
            object : ReverseTensor(newPrimal, x.derivativeID) {
                override fun backpropagate() {
                    val originalGradientShape = x.shape + derivativeID.upstreamShape
                    x.pushback(
                        org.diffkt.zeroOfSameKind(newPrimal, originalGradientShape).withChange(index, axis, upstream))
                }
            }
    }

    override fun view3(x: DTensor, index: IntRange, axis: Int): DTensor {
        require(x is ReverseTensor)
        val newPrimal = x.primal.view(index, axis)
        return object : ReverseTensor(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                val originalGradientShape = x.shape + derivativeID.upstreamShape
                x.pushback(org.diffkt.zeroOfSameKind(newPrimal, originalGradientShape).withChange(index, axis, upstream))
            }
        }
    }

    override fun reshape(x: DTensor, newShape: Shape): DTensor {
        require(x is ReverseTensor)
        // For ReverseTensor, we make a recursive reshape call for
        // the primal, but for the gradient we create a ReverseTensor with
        // machinery for backprop as well as the pushback logic. See notes
        // on pushback logic below.
        val newPrimal = x.primal.operations.reshape(x.primal, newShape)
        return object : ReverseTensor(newPrimal, x.derivativeID)
        {
            override fun backpropagate() {
                // upstreamShape is the shape of the output of the
                // function we are taking the derivative of. We have to return
                // the derivative of each of the outputs of the function with
                // respect to each element of outTensor, so our gradient shapes
                // need to be `oldTensor.shape + derivativeID.upstreamShape`.
                //
                // Note that if oldTensor is the input to the function we are
                // taking the derivative of, then the gradient shape will be
                // `functionInputShape concat functionOutputShape`, and we will
                // have the gradient of each output with respect to each input.
                // Concrete examples of this can be found in FlattenTest.
                val g = upstream.reshape(x.shape + derivativeID.upstreamShape)
                x.pushback(g)
            }
        }
    }

    override fun reshapeToScalar(x: DTensor): DScalar {
        require(x is ReverseTensor)
        val newPrimal = x.primal.operations.reshapeToScalar(x.primal)
        return object : ReverseScalar(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                val g = upstream.reshape(x.shape + derivativeID.upstreamShape)
                x.pushback(g)
            }
        }
    }

    override fun squeeze(x: DTensor, axis: Int): DTensor {
        require(x is ReverseTensor)
        val newPrimal = x.primal.squeeze(axis)
        return object : ReverseTensor(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream.unsqueeze(axis))
            }
        }
    }

    override fun unsqueeze(x: DTensor, axis: Int): DTensor {
        require(x is ReverseTensor)
        val newPrimal = x.primal.unsqueeze(axis)
        return object : ReverseTensor(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream.squeeze(axis))
            }
        }
    }

    override fun transpose(x: DTensor, axes: IntArray): DTensor {
        require(x is ReverseTensor)
        fun invertAndExtendPermutation(axes: IntArray, additional: Int): IntArray {
            val result = IntArray(axes.size + additional)
            for (i in axes.indices)
                result[axes[i]] = i
            for (i in axes.size until result.size)
                result[i] = i
            return result
        }
        val newPrimal = x.primal.transpose(axes)
        return object : ReverseTensor(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream.transpose(invertAndExtendPermutation(axes, derivativeID.upstreamShape.rank)))
            }
        }
    }

    override fun relu(x: DTensor): DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(x.primal.relu(), x.derivativeID) {
            override fun backpropagate() {
                val grad = reluGrad(x.primal, upstream)
                x.pushback(grad)
            }
        }
    }

    override fun reluGrad(x: DTensor, reluUpstream: DTensor, derivativeId: DerivativeID): DTensor {
        require(derivativeId is ReverseDerivativeID)
        val up = wrap(reluUpstream, derivativeId)
        return object : ReverseTensor(reluGrad(x.primal, up.primal), derivativeId) {
            override fun backpropagate() {
                // TODO: should there be a call to x.pushback(...) ?
                val upd = reluGrad(x.primal, upstream)
                up.pushback(upd)
            }
        }
    }

    override fun sigmoid(x: DTensor): DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(org.diffkt.sigmoid(x.primal), x.derivativeID) {
            override fun backpropagate() {
                val derivative = (primal * (1f - primal)).expandToTangent(upstream)
                x.pushback(upstream * derivative)
            }
        }
    }

    override fun sum(x: DTensor, axes: IntArray, keepDims: Boolean): DTensor {
        require(x is ReverseTensor)
        val newPrimal = x.primal.sum(axes, keepDims)
        return if (newPrimal is DScalar)
            object : ReverseScalar(newPrimal, x.derivativeID) {
                override fun backpropagate() {
                    var currTens = upstream
                    if (!keepDims) {
                        for (axis in axes.sorted())
                            currTens = currTens.unsqueeze(axis)
                    }
                    val g = currTens.expand(x.shape + derivativeID.upstreamShape)
                    x.pushback(g)
                }
            }
        else
            object : ReverseTensor(newPrimal, x.derivativeID) {
                override fun backpropagate() {
                    var currTens = upstream
                    if (!keepDims) {
                        for (axis in axes.sorted())
                            currTens = currTens.unsqueeze(axis)
                    }
                    val g = currTens.expand(x.shape + derivativeID.upstreamShape)
                    x.pushback(g)
                }
            }
    }

    override fun avgPool(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        require(x is ReverseTensor)
        val newPrimal = org.diffkt.model.avgPool(x.primal, poolHeight, poolWidth)
        return object : ReverseTensor(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                val grad = org.diffkt.model.avgPoolGrad(upstream, poolHeight, poolWidth)
                x.pushback(grad)
            }
        }
    }

    override fun avgPoolGrad(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        require(x is ReverseTensor)
        val newPrimal = org.diffkt.model.avgPoolGrad(x.primal, poolHeight, poolWidth)
        return object : ReverseTensor(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                val grad = org.diffkt.model.avgPool(upstream, poolHeight, poolWidth)
                x.pushback(grad)
            }
        }
    }

    override fun batchNorm(input: DTensor, scaleShift: DTensor, derivativeId: DerivativeID): BatchNormResult {
        require(derivativeId is ReverseDerivativeID)
        if (input.rank == 4
            && isDnnlEligible(input)
            && isDnnlEligible(scaleShift)
            && input.derivativeID.sequence == scaleShift.derivativeID.sequence
        ) {
            scaleShift as ReverseTensor
            input as ReverseTensor
            val inputPrimal = input.primal
            val scaleShiftPrimal = scaleShift.primal
            require(inputPrimal is FloatTensor && scaleShiftPrimal is FloatTensor)

            val primalResult = org.diffkt.model.batchNorm(inputPrimal, scaleShiftPrimal)
            primalResult.mean as FloatTensor
            primalResult.variance as FloatTensor
            val newPrimal = primalResult.result
            val res = object : ReverseTensor(newPrimal, input.derivativeID) {
                override fun backpropagate() {
                    require(derivativeID.upstreamShape == Shape()) {
                        "Fast batchnorm gradients are only supported for functions that return a scalar" }
                    val upstreamT = upstream
                    require(upstreamT is StridedFloatTensor) { "Higher order batchNorm gradient not supported" }
                    val (inputGrad, scaleShiftGrad) = Dnnl.batchNormGrad(
                        upstreamT,
                        inputPrimal,
                        scaleShiftPrimal,
                        primalResult.mean,
                        primalResult.variance
                    )
                    input.pushback(inputGrad)
                    scaleShift.pushback(scaleShiftGrad)
                }
            }
            return BatchNormResult.fromMeanAndVariance(res, primalResult.mean, primalResult.variance)
        }

        return baseBatchNorm(input, scaleShift)
    }

    private fun maxpoolDnnlGrad(
        x: DTensor,
        poolHeight: Int,
        poolWidth: Int,
        maxIndices: ByteArray,
        seed: DTensor
    ): FloatTensor {
        require(seed is FloatTensor) { TODO("Dnnl maxPool higher order grad not supported") }
        x as StridedFloatTensor
        val seedN = seed.normalize()
        val outStream = FloatArray(x.data.size)
        Dnnl.maxPoolGrad(
            // result
            x.shape.dims,
            outStream,
            // max indices calculated during forward
            maxIndices,
            // seed
            seedN.shape.dims,
            seedN.data,
            // pool height and width
            poolHeight,
            poolWidth
        )
        return FloatTensor(x.shape, outStream)
    }
    override fun maxPoolWithIndices(
        x: DTensor,
        poolHeight: Int,
        poolWidth: Int,
        withIndices: Boolean
    ): Pair<DTensor, List<IntArray>?> {
        require(x is ReverseTensor)
        val primal = x.primal
        if (primal is StridedFloatTensor && shouldSendToCpp(100, Dnnl, primal) && !withIndices) {
            val (newPrimal, indices) = maxPoolWithIndicesDnnl(primal, poolHeight, poolWidth, withIndices = true)
            indices!!
            val result = object : ReverseTensor(newPrimal, x.derivativeID) {
                override fun backpropagate() {
                    x.pushback(maxpoolDnnlGrad(x.primal, poolHeight, poolWidth, indices, upstream))
                }
            }
            return Pair(result, null)
        }
        val (newPrimal, indices) = x.primal.operations.maxPoolWithIndices(
            x.primal,
            poolHeight,
            poolWidth,
            withIndices = true
        )
        val result = object : ReverseTensor(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream.operations.scatterAtIndices(upstream, indices!!, x.shape + x.derivativeID.upstreamShape))
            }
        }
        return Pair(result, indices)
    }

    override fun gather(x: DTensor, indices: List<Int>, axis: Int, paddingIndex: Int): DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(x.primal.operations.gather(x.primal, indices, axis, paddingIndex), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream.operations.scatter(upstream, indices, axis, x.shape + x.derivativeID.upstreamShape, paddingIndex))
            }
        }
    }

    override fun gatherAtIndices(x: DTensor, indices: List<IntArray>): DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(x.primal.operations.gatherAtIndices(x.primal, indices), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream.operations.scatterAtIndices(upstream, indices, x.shape + x.derivativeID.upstreamShape))
            }
        }
    }

    override fun scatter(x: DTensor, indices: List<Int>, axis: Int, newShape: Shape, paddingIndex: Int): DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(x.primal.operations.scatter(x.primal, indices, axis, newShape, paddingIndex), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream.operations.gather(upstream, indices, axis, paddingIndex))
            }
        }
    }

    override fun scatterAtIndices(x: DTensor, indices: List<IntArray>, newShape: Shape): DTensor {
        require(x is ReverseTensor)
        return object : ReverseTensor(x.primal.operations.scatterAtIndices(x.primal, indices, newShape), x.derivativeID) {
            override fun backpropagate() {
                x.pushback(upstream.operations.gatherAtIndices(upstream, indices))
            }
        }
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
        throw IllegalStateException("Should not get here")
    }

    @SType("S: Shape")
    override fun ifThenElse(
        condition: @SType("S") DTensor,
        whenTrue: @SType("S") DTensor,
        whenFalse: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(derivativeId is ReverseDerivativeID)
        val l = wrap(whenTrue, derivativeId)
        val r = wrap(whenFalse, derivativeId)
        val primal = ifThenElse(condition, l.primal, r.primal)
        return object : ReverseTensor(primal, derivativeId) {
            override fun backpropagate() {
                if (condition is FloatScalar) {
                    if (condition.value > 0f) {
                        l.pushback(upstream)
                    } else {
                        r.pushback(upstream)
                    }
                } else {
                    val pred = if (condition is DScalar) condition else condition.expandAndBroadcastToTangent(upstream)
                    val zeros = primal.operations.zeroOfSameKind(primal, upstream.shape)
                    l.pushback(ifThenElse(pred, upstream, zeros))
                    r.pushback(ifThenElse(pred, zeros, upstream))
                }
            }
        }
    }
}
