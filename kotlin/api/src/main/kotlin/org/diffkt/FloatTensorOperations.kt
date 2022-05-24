/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.external.*
import org.diffkt.model.BatchNormResult
import org.diffkt.model.baseBatchNorm
import org.diffkt.random.RandomKey
import org.diffkt.random.Sha512Random
import kotlin.math.ceil
import kotlin.math.pow
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

/**
 * Operations that can be shared among multiple float tensor implementations.
 */
internal abstract class FloatTensorOperations : Operations {
    @SType("S: Shape")
    private fun wrap(value: @SType("S") DTensor): @SType("S") FloatTensor {
        if (value is FloatTensor) return value
        TODO("Cannot (automatically) convert to FloatTensor")
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
        return l.zip(r) { x, y -> x + y }
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
        return l.zip(r) { x, y -> x - y }
    }

    @SType("S: Shape")
    override fun times(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(derivativeId == NoDerivativeID)
        return wrap(left).zip(wrap(right)) { l, r -> l * r }
    }

    @SType("S: Shape")
    override fun timesScalar(left: DScalar, right: @SType("S") DTensor, derivativeId: DerivativeID): @SType("S") DTensor {
        require(derivativeId == NoDerivativeID)
        val r = wrap(right)
        require(left is FloatScalar)
        val leftValue = left.value
        return r.map { x -> leftValue * x }
    }

    @SType("S: Shape")
    override fun div(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(left is FloatTensor)
        require(right is FloatTensor)
        return left.zip(right) { xx, yy -> xx / yy }
    }

    @SType("S: Shape")
    override fun zeroOfSameKind(x: DTensor, shape: @SType("S") Shape): @SType("S") FloatTensor {
        return FloatTensor.zeros(shape)
    }

    @SType("S: Shape")
    @AllowUnreduced
    override fun identityGradientOfSameKind(x: DTensor, halfShape: @SType("S") Shape): @SType("concat(S,S)") FloatTensor {
        return StridedFloatTensor.identityGradient(halfShape)
    }

    @SType("S: Shape")
    override fun unaryMinus(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { -it }
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
        val left = (x as FloatTensor).asStrided()
        val right = (y as FloatTensor).asStrided()
        require(c.rank != 0)
        if (a.rank < 12 && b.rank == 1 && c.rank == 1 && d.rank == 1)
            return Dnnl.matmul(left, right, a, b, d)

        fun processC(axis: Int, leftOffset: Int, rightOffset: Int): Float {
            assert(axis < c.rank)
            val leftStride = left.strides[a.rank + b.rank + axis]
            val rightStride = right.strides[a.rank + axis]
            var leftOff = leftOffset
            var rightOff = rightOffset
            var sum = 0F
            if (axis == c.rank - 1) {
                // This is the inner loop.  Can Dnnl help?
                for (i in 0 until c[axis]) {
                    sum += left.data[leftOff] * right.data[rightOff]
                    leftOff += leftStride
                    rightOff += rightStride
                }
            } else {
                for (i in 0 until c[axis]) {
                    sum += processC(axis + 1, leftOff, rightOff)
                    leftOff += leftStride
                    rightOff += rightStride
                }
            }

            return sum
        }

        val resultShape = a + b + d
        val data = FloatArray(resultShape.product)
        var next = 0

        fun processD(axis: Int, leftOffset: Int, rightOffset: Int) {
            assert(axis <= d.rank)
            if (d.rank == 0) {
                data[next++] = processC(0, leftOffset, rightOffset)
                return
            }
            val rightStride = right.strides[a.rank + c.rank + axis]
            var rightOff = rightOffset
            if (axis == d.rank - 1) {
                for (i in 0 until d[axis]) {
                    data[next++] = processC(0, leftOffset, rightOff)
                    rightOff += rightStride
                }
            } else {
                for (i in 0 until d[axis]) {
                    processD(axis + 1, leftOffset, rightOff)
                    rightOff += rightStride
                }
            }
        }

        fun processB(axis: Int, leftOffset: Int, rightOffset: Int) {
            if (axis >= b.rank) {
                processD(0, leftOffset, rightOffset)
                return
            }

            val leftStride = left.strides[a.rank + axis]
            var leftOff = leftOffset
            for (i in 0 until b[axis]) {
                processB(axis + 1, leftOff, rightOffset)
                leftOff += leftStride
            }
        }

        fun processA(axis: Int, leftOffset: Int, rightOffset: Int) {
            if (axis >= a.rank) {
                processB(0, leftOffset, rightOffset)
                return
            }

            val leftStride = left.strides[axis]
            val rightStride = right.strides[axis]
            var leftOff = leftOffset
            var rightOff = rightOffset
            for (i in 0 until a[axis]) {
                processA(axis + 1, leftOff, rightOff)
                leftOff += leftStride
                rightOff += rightStride
            }
        }

        processA(0, left.offset, right.offset)
        assert(next == data.size)
        return FloatTensor(resultShape, data)
    }

    @SType("S1: Shape, S2: Shape")
    @AllowUnreduced
    override fun outerProduct(
        x: @SType("S1") DTensor,
        y: @SType("S2") DTensor,
        derivativeId: DerivativeID
    ): @SType("concat(S1, S2)") DTensor {
        val left = (x as @SType("S1") FloatTensor).asStrided()
        val right = (y as @SType("S2") FloatTensor).asStrided()
        val resultData = FloatArray(left.size * right.size)
        var k = 0
        for (i in 0 until left.size) {
            val l = left.at(i)
            for (j in 0 until right.size) {
                val r = right.at(j)
                resultData[k++] = l * r
            }
        }
        return FloatTensor(left.shape + right.shape, resultData)
    }

    @SType("S: Shape")
    override fun sin(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { kotlin.math.sin(it) }
    }

    @SType("S: Shape")
    override fun cos(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { kotlin.math.cos(it) }
    }

    @SType("S: Shape")
    override fun tan(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { kotlin.math.tan(it) }
    }

    @SType("S: Shape")
    override fun atan(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { kotlin.math.atan(it) }
    }

    @SType("S: Shape")
    override fun exp(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { kotlin.math.exp(it) }
    }

    @SType("S: Shape")
    override fun ln(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { kotlin.math.ln(it) }
    }

    @SType("S: Shape")
    override fun lgamma(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { Math.lgamma(it) }
    }

    @SType("S: Shape")
    override fun digamma(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { Math.digamma(it) }
    }

    @SType("S: Shape")
    override fun polygamma(n: Int, x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { Math.polygamma(n, it) }
    }

    @SType("S: Shape")
    override fun sqrt(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { kotlin.math.sqrt(it) }
    }

    @SType("S: Shape")
    override fun tanh(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { kotlin.math.tanh(it) }
    }

    override fun meld(values: List<DTensor>, derivativeId: DerivativeID): DTensor {
        require(derivativeId == NoDerivativeID)
        // Compute the sizes of the values
        val totalSize = values.map { it.shape.product() }.sum()
        val serializedData = FloatArray(totalSize)
        var i = 0
        for (v in values) {
            when (v) {
                is FloatScalar ->
                    serializedData[i++] = v.value
                is FloatTensor -> {
                    for (pos in 0 until v.size) serializedData[i++] = v.at(pos)
                }
                else -> throw IllegalArgumentException("meld not supported for ${v::class.qualifiedName}")
            }
        }
        assert(i == totalSize)
        return FloatTensor(Shape(totalSize), serializedData)
    }

    override fun split(x: DTensor, shapes: List<Shape>): List<DTensor> {
        require(x is FloatTensor)
        val sizes = shapes.map { it.product() }
        var nextStart = 0
        return List(shapes.size) {
            val shape = shapes[it]
            val size = sizes[it]
            val partData = FloatArray(size) { i -> x.at(i + nextStart) }
            val part = FloatTensor(shape, partData)
            nextStart += size
            part
        }
    }

    @AllowUnreduced
    @SType("S1: Shape, S2: Shape, A: Dim")
    override fun concat(
        left: @SType("S1")  DTensor,
        right: @SType("S2") DTensor,
        axis: @SType("A") Int,
        derivativeId: DerivativeID
    ): @SType("concatOnAxis(S1, S2, A)") DTensor {
        require(derivativeId == NoDerivativeID)
        val l = (left as FloatTensor).asStrided()
        val r = (right as FloatTensor).asStrided()
        return concat(listOf(l, r), axis, derivativeId)
    }

    override fun concat(slices: List<DTensor>, axis: Int, derivativeId: DerivativeID): DTensor {
        /** Copy a value to a section of a destination array */
        fun fillDim(
            value: Float,
            dest: FloatArray,
            destShape: Shape,
            destStart: Int,
            len: Int,
            axis: Int,
            skipZero: Boolean
        ) {
            if (skipZero && value == 0f) return

            val nframes = destShape.take(axis).product()
            val cellSize = destShape.drop(axis + 1).product()
            val destFrameSize = destShape[axis]

            for (i in 0 until nframes) {
                val dstOffset = ((i * destFrameSize) + destStart) * cellSize
                for (cellOffset in 0 until cellSize * len) {
                    dest[dstOffset + cellOffset] = value
                }
            }
        }

        /**
         * copy some portion of a tensor data array along a particular dimension
         * to a destination array.
         * NOTE: assumes src Tensor has Natural (contiguous, full) layout
         */
        fun copyDim(
            src: FloatArray,
            srcShape: Shape,
            srcStart: Int,
            dest: FloatArray,
            destShape: Shape,
            destStart: Int,
            len: Int,
            axis: Int
        ): FloatArray {
            val nframes = destShape.take(axis).product()
            val cellSize = destShape.drop(axis + 1).product()
            val srcFrameSize = srcShape[axis]
            val destFrameSize = destShape[axis]

            var i = 0
            while (i < nframes) {
                val srcOff = ((i * srcFrameSize) + srcStart) * cellSize
                val dstOff = ((i * destFrameSize) + destStart) * cellSize
                System.arraycopy(src, srcOff, dest, dstOff, cellSize * len)
                i += 1
            }
            return dest
        }

        /** helper, calls [copyDim] or [fillDim] as needed to copy source tensor data to dest
         * as specified.
         * Note: may copy src tensor to a contiguous layout version, in which case that tensor
         * is returned, otherwise src itself.
         *
         * @param skipZero doesn't copy a singleton zero over to the dst array.
         *   This should be used as an optimization when the dst array was already
         *   initialized to zero.
         */
        fun copyDimFromTensor(
            src: StridedFloatTensor,
            srcStart: Int,
            dest: FloatArray,
            destShape: Shape,
            destStart: Int,
            len: Int,
            axis: Int,
            skipZero: Boolean = false
        ): FloatTensor {
            return if (src.layout == StridedUtils.Layout.SINGLETON) {
                fillDim(src.data[src.offset], dest, destShape, destStart, len, axis, skipZero)
                src
            } else {
                val srcContig = src.normalize()
                copyDim(srcContig.data, srcContig.shape, srcStart, dest, destShape, destStart, len, axis)
                srcContig
            }
        }

        require(derivativeId == NoDerivativeID)
        require(slices.size > 0)
        val stridedSlices = slices.map { (it as FloatTensor).asStrided() }
        val first = stridedSlices[0]
        require(axis >= 0 && axis < first.rank)
        if (stridedSlices.size == 1)
            return first
        val s = first.shape.updated(axis, 1)
        require(stridedSlices.all { first.rank == it.rank && s == it.shape.updated(axis, 1) })

        val newDim = stridedSlices.map { it.shape[axis] }.sum()
        val newShape = s.updated(axis, newDim)
        val newData = FloatArray(newShape.product)
        var next = 0
        for (slice in stridedSlices) {
            val amount = slice.shape[axis]
            copyDimFromTensor(slice, 0, newData, newShape, next, amount, axis, skipZero = true)
            next += amount
        }
        return FloatTensor(newShape, newData)
    }

    @SType("S: Shape")
    override fun broadcastTo(x: DTensor, newShape: @SType("S") Shape): @SType("S") DTensor {
        require(x is FloatTensor)
        val s = x.asStrided()
        val (newStrides, _) = Broadcasting.getBroadcastedStridesAndAxes(s.shape, s.strides, newShape)
        return StridedFloatTensor(newShape, offset = s.offset, newStrides, s.data)
    }

    override fun convImpl(
        signal: DTensor,
        filter: DTensor,
        hStride: Int,
        vStride: Int,
        padding: Convolve.Padding2D,
        derivativeId: DerivativeID
    ): DTensor {
        require(signal is FloatTensor)
        require(filter is FloatTensor)
        val sN = signal.normalize()
        val fN = filter.normalize()

        require(shouldSendToCpp(0, Dnnl, sN, fN))
        val signalShape = sN.shape
        val filterShape = fN.shape

        val imageChannels = signalShape[3]
        val filterChannels = filterShape[3]

        if (imageChannels != filterChannels)
            throw RuntimeException("the size of the filter's inChannel ($filterChannels) must match the input depth ($imageChannels)")
        if (hStride < 1 || vStride < 1)
            throw RuntimeException("Horizontal stride ($hStride) and vertical stride ($vStride) must be greater than 0.")

        // Make out shape
        val numsignal = signalShape[Convolve.N_AXIS]
        val numfilter = filterShape[Convolve.N_AXIS]

        val endRow = signalShape[Convolve.H_AXIS] + padding.bottom - filterShape[Convolve.H_AXIS]
        val endCol = signalShape[Convolve.W_AXIS] + padding.right - filterShape[Convolve.W_AXIS]

        val outHeight = ceil((endRow + padding.top + 1).toFloat() / vStride).toInt()
        val outWidth = ceil((endCol + padding.left + 1).toFloat() / hStride).toInt()

        val outShape = Shape(numsignal, outHeight, outWidth, numfilter)
        // End make out shape

        return StridedFloatTensor.contiguous(outShape) {
            Dnnl.conv2d(
                // output
                outShape.dims,
                it,
                // imgs
                signalShape.dims,
                sN.data,
                // filter
                filterShape.dims,
                fN.data,
                // strides
                vStride, // height
                hStride, // width
                // padding
                padding.left,
                padding.right,
                padding.top,
                padding.bottom
            )
        }
    }

    /**
     * Helper function to do the stride calculations for expand.
     *
     * @return strides for the return value of expand
     */
    private fun expandStrides(x: StridedFloatTensor, shape: Shape): IntArray {
        require(x.rank == shape.rank)
        return IntArray(x.rank) { i ->
            if (x.shape[i] != shape[i]) 0 else x.strides[i]
        }
    }
    override fun expand(x: DTensor, newShape: Shape): DTensor {
        require(x is FloatTensor)
        val s = x.asStrided()
        val newStrides = expandStrides(s, newShape)
        return StridedFloatTensor(newShape, s.offset, newStrides, s.data)
    }

    /**
     * A copy-free implementation of flip for strided float tensors.
     */
    private fun flip(x: StridedFloatTensor, axis: Int): StridedFloatTensor {
        require(axis >= 0 && axis < x.rank)
        if (x.shape[axis] == 1 || x.strides[axis] == 0) return x
        val newOffset = x.offset + (x.shape[axis] - 1) * x.strides[axis]
        val newStrides = x.strides.clone(); newStrides[axis] = -x.strides[axis]
        return StridedFloatTensor(x.shape, newOffset, newStrides, x.data)
    }

    @SType("S: Shape")
    override fun flip(x: @SType("S") DTensor, axes: IntArray): @SType("S") DTensor {
        require(x is FloatTensor)
        return axes.fold(x.asStrided()) { a, i -> flip(a, i) }
    }

    override fun transpose(x: DTensor, axes: IntArray): DTensor {
        require(x is FloatTensor)
        val n = x.asStrided()
        return n.operations.transpose(n, axes)
    }

    override fun logSoftmax(x: DTensor, axis: Int): DTensor {
        require(x is FloatTensor)
        val normalized = x.normalize()
        val resData = FloatArray(normalized.size)
        Dnnl.logSoftmax(normalized.shape.dims, normalized.data, resData, axis)
        return FloatTensor(normalized.shape, resData)
    }

    override fun logSoftmaxGrad(x: DTensor, axis: Int, logSoftmax: DTensor, upstream: DTensor): DTensor {
        require(upstream.shape == x.shape) {
            "LogSoftmax does not support derivatives of functions that do not return scalars"
        }
        val normalUpstream = wrap(upstream).normalize()
        val normalLogSoftmax = wrap(logSoftmax).normalize()
        val gradData = FloatArray(x.size)
        Dnnl.logSoftmaxGrad(x.shape.dims, gradData, normalUpstream.data, normalLogSoftmax.data, axis)
        return FloatTensor(x.shape, gradData)
    }

    @SType("S: Shape")
    override fun pow(base: @SType("S") DTensor, exponent: Float): @SType("S") DTensor {
        require(base is FloatTensor)
        return base.map { it.pow(exponent) }
    }

    override fun view1(x: DTensor, indices: IntArray): DTensor {
        require(x is FloatTensor)
        return StridedFloatTensorOperations.view1(x.asStrided(), indices)
    }

    override fun view2(x: DTensor, index: Int, axis: Int): DTensor {
        require(x is FloatTensor)
        return StridedFloatTensorOperations.view2(x.asStrided(), index, axis)
    }

    override fun view3(x: DTensor, index: IntRange, axis: Int): DTensor {
        require(x is FloatTensor)
        return StridedFloatTensorOperations.view3(x.asStrided(), index, axis)
    }

    override fun reshape(x: DTensor, newShape: Shape): DTensor {
        require(x is FloatTensor)
        val n = x.normalize()
        return n.operations.reshape(n, newShape)
    }

    override fun reshapeToScalar(x: DTensor): DScalar {
        require(x is FloatTensor)
        return FloatScalar(x.at(0))
    }

    override fun squeeze(x: DTensor, axis: Int): DTensor {
        require(x is FloatTensor)
        return StridedFloatTensorOperations.squeeze(x.asStrided(), axis)
    }

    override fun unsqueeze(x: DTensor, axis: Int): DTensor {
        require(x is FloatTensor)
        return StridedFloatTensorOperations.unsqueeze(x.asStrided(), axis)
    }

    @SType("S: Shape")
    override fun relu(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { v -> if (v <= 0F) 0f else v }
    }

    override fun reluGrad(x: DTensor, reluUpstream: DTensor, derivativeId: DerivativeID): DTensor {
        val xx = x.expandAndBroadcastToTangent(reluUpstream)
        require(xx is FloatTensor)
        require(reluUpstream is FloatTensor)
        return xx.zip(reluUpstream) { x1, up1 -> if (x1 <= 0F) 0f else up1 }
    }

    @SType("S: Shape")
    override fun sigmoid(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is FloatTensor)
        return x.map { sigmoidElem(it) }
    }

    override fun sum(x: DTensor, axes: IntArray, keepDims: Boolean): DTensor {
        require(x is FloatTensor)
        return x.reduce(Float::plus, axes, keepDims)
    }

    override fun avgPool(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        require(x is FloatTensor)
        val numItems = x.shape[Convolve.N_AXIS]
        val inHeight = x.shape[Convolve.H_AXIS]
        val inWidth = x.shape[Convolve.W_AXIS]
        val channels = x.shape.drop(Convolve.C_AXIS)
        val numChannels = channels.product

        require(inHeight % poolHeight == 0) {
            "input height ($inHeight) must be divisible by pool height ($poolHeight)" }
        require(inWidth % poolWidth == 0) {
            "input width ($inWidth) must be divisible by pool width ($poolWidth)" }

        val outHeight = inHeight / poolHeight
        val outWidth = inWidth / poolWidth

        val outShape = Shape(numItems, outHeight, outWidth) + channels
        val outStream = FloatArray(outShape.product)
        Dnnl.avgPool(
            // result
            intArrayOf(numItems, outHeight, outWidth, numChannels),
            outStream,
            // input
            intArrayOf(numItems, inHeight, inWidth, numChannels),
            x.normalize().data,
            // pool height and width
            poolHeight,
            poolWidth
        )

        return FloatTensor(outShape, outStream)
    }

    override fun avgPoolGrad(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor {
        require(x is FloatTensor)
        val numItems = x.shape[Convolve.N_AXIS]
        val inHeight = x.shape[Convolve.H_AXIS]
        val inWidth = x.shape[Convolve.W_AXIS]
        val channels = x.shape.drop(Convolve.C_AXIS)
        val numChannels = channels.product
        val outHeight = inHeight * poolHeight
        val outWidth = inWidth * poolWidth

        val outShape = Shape(numItems, outHeight, outWidth) + channels
        val outStream = FloatArray(outShape.product)
        Dnnl.avgPoolGrad(
            // result
            intArrayOf(numItems, outHeight, outWidth, numChannels),
            outStream,
            // seed
            intArrayOf(numItems, inHeight, inWidth, numChannels),
            x.normalize().data,
            // pool height and width
            poolHeight,
            poolWidth
        )
        return FloatTensor(outShape, outStream)
    }

    override fun batchNorm(input: DTensor, scaleShift: DTensor, derivativeId: DerivativeID): BatchNormResult {
        return baseBatchNorm(input, scaleShift)
    }

    override fun maxPoolWithIndices(
        x: DTensor,
        poolHeight: Int,
        poolWidth: Int,
        withIndices: Boolean
    ): Pair<DTensor, List<IntArray>?> {
        require(x is FloatTensor)
        val N_AXIS = 0
        val H_AXIS = 1
        val W_AXIS = 2
        val C_AXIS = 3

        val numItems = x.shape[N_AXIS]
        val inHeight = x.shape[H_AXIS]
        val inWidth = x.shape[W_AXIS]
        val numChannels = x.shape[C_AXIS]
        require(inHeight % poolHeight == 0) {
            "input height ($inHeight) must be divisible by pool height ($poolHeight)" }
        require(inWidth % poolWidth == 0) {
            "input width ($inWidth) must be divisible by pool width ($poolWidth)" }
        val outHeight = inHeight / poolHeight
        val outWidth = inWidth / poolWidth
        val outShape = Shape(numItems, outHeight, outWidth, numChannels)

        val values = FloatArray(outShape.product)
        var nextValuePos = 0
        val positions: ArrayList<IntArray>? = if (withIndices) ArrayList<IntArray>() else null

        val indices = IntArray(4)
        for (item in 0 until numItems) {
            indices[N_AXIS] = item
            for (h0 in 0 until inHeight step poolHeight) {
                for (w0 in 0 until inWidth step poolWidth) {
                    for (channel in 0 until numChannels) {
                        indices[C_AXIS] = channel
                        var maxH = 0
                        var maxW = 0
                        var maxValue = Float.NEGATIVE_INFINITY
                        for (h1 in 0 until poolHeight) {
                            for (w1 in 0 until poolHeight) {
                                val h = h0 + h1
                                val w = w0 + w1
                                indices[H_AXIS] = h
                                indices[W_AXIS] = w
                                val value = x.getAt(indices)
                                if (value > maxValue) {
                                    maxH = h
                                    maxW = w
                                    maxValue = value
                                }
                            }
                        }
                        indices[H_AXIS] = maxH
                        indices[W_AXIS] = maxW
                        values[nextValuePos++] = maxValue
                        positions?.add(indices.clone())
                    }
                }
            }
        }

        return Pair(FloatTensor(outShape, values), positions)
    }

    override fun gather(x: DTensor, indices: List<Int>, axis: Int, paddingIndex: Int): DTensor {
        require(x is FloatTensor)

        val rank = x.rank
        require(rank > 0) { "gather: tensor must be of rank > 0" }
        require(axis >= 0) { "gather: axis $axis < 0" }
        require(axis < rank) { "gather: axis $axis >= tensor rank $rank" }
        require(indices.all { it < x.shape[axis] })

        val newShape = x.shape.updated(axis, indices.size)
        val newData = FloatArray(newShape.product)

        val numOfIterations = x.shape.take(axis).product
        val elementsPerIteration = x.shape.drop(axis + 1).product
        val stride = x.shape.drop(axis).product

        var idx = 0
        for (n in 0 until numOfIterations) {
            for (i in indices) {
                if (i == paddingIndex) {
                    idx += elementsPerIteration
                } else {
                    val startingPoint = (n * stride) + (i * elementsPerIteration)
                    for (e in 0 until elementsPerIteration) {
                        newData[idx] = x.at(startingPoint + e)
                        idx++
                    }
                }
            }
        }
        return FloatTensor(newShape, newData)
    }

    override fun gatherAtIndices(x: DTensor, indices: List<IntArray>): DTensor {
        require(x is FloatTensor)
        require(indices.all { it.size == indices[0].size })
        // TODO: https://github.com/facebookincubator/diffkt/issues/89 Dnnl could probably help us here
        val shapePerIndex = x.shape.drop(indices[0].size)
        val elementsPerIndex = shapePerIndex.product
        val totalSize = indices.size * elementsPerIndex
        val newData = FloatArray(totalSize)
        val contigStrides = StridedUtils.contigStrides(x.shape)
        for (i in indices.indices) {
            val index = indices[i]
            val destPos = i * elementsPerIndex
            val srcPos = index.foldIndexed(0, { elementIndex, accum, elem -> accum + elem*contigStrides[elementIndex] })
            for (j in 0 until elementsPerIndex) {
                newData[destPos + j] = x.at(srcPos + j)
            }
        }
        val newShape = shapePerIndex.prepend(indices.size)
        assert(newShape.product == newData.size)
        return FloatTensor(newShape, newData)
    }

    override fun scatter(x: DTensor, indices: List<Int>, axis: Int, newShape: Shape, paddingIndex: Int): DTensor {
        require(x is FloatTensor)

        val rank = x.rank
        require(rank > 0) { "scatter: tensor must be of rank > 0" }
        require(axis >= 0) { "scatter: axis $axis < 0" }
        require(axis < rank) { "scatter: axis $axis >= tensor rank $rank" }
        require(indices.all { it < newShape[axis] })

        if (axis == 0 && newShape.rank == 2) {
            return scatterSparseRow(x, indices, newShape, paddingIndex)
        } else {
            return scatterDense(x, indices, axis, newShape, paddingIndex)
        }

    }

    private fun scatterDense(x: FloatTensor, indices: List<Int>, axis: Int, newShape: Shape, paddingIndex: Int): FloatTensor {
        val newData = FloatArray(newShape.product)

        val numOfIterations = x.shape.take(axis).product
        val elementsPerIteration = x.shape.drop(axis + 1).product
        val xStride = x.shape.drop(axis).product
        val newDataStride = newShape.drop(axis).product

        for (n in 0 until numOfIterations) {
            val xStartingPoint = n * xStride
            val newDataStartingPoint = n * newDataStride
            indices.forEachIndexed { i, index ->
                if (index == paddingIndex) return@forEachIndexed
                for (e in 0 until elementsPerIteration) {
                    val newIndex = newDataStartingPoint + (index * elementsPerIteration) + e
                    newData[newIndex] += x.at(xStartingPoint+ (i * elementsPerIteration) + e)
                }
            }
        }
        return FloatTensor(newShape, newData)
    }

    private fun scatterSparseRow(x: FloatTensor, indices: List<Int>, newShape: Shape, paddingIndex: Int): FloatTensor {
        val sortedIndices = indices.withIndex().sortedBy { it.value }
        var numUnique = 0
        var prev = -1
        for (i in sortedIndices) {
            if (i.value != prev && i.value != paddingIndex) {
                numUnique++
                prev = i.value
            }
        }
        if (numUnique == newShape[0]) return scatterDense(x, indices, 0, newShape, paddingIndex)
        val tableWidth = x.shape.last
        require(tableWidth == newShape[1]) { "scatterSparseRow: different number of columns not yet supported"}
        val dataWidth = newShape[1] + 1
        val newData = FloatArray(dataWidth * numUnique)

        prev = -1
        var newRowIndex = 0
        sortedIndices.forEach { i ->
            if (i.value == paddingIndex) return@forEach
            if (prev != -1 && i.value != prev) {
                newRowIndex++
            }
            val offset = newRowIndex * dataWidth
            newData[offset] = i.value.toFloat()

            for (j in 1 until dataWidth) {
                newData[offset + j] += x.at(tableWidth * i.index + j - 1)
            }
            prev = i.value
        }
        return SparseRowFloatTensor(newShape, newData)
    }

    override fun scatterAtIndices(x: DTensor, indices: List<IntArray>, newShape: Shape): DTensor {
        require(x is FloatTensor)
        // TODO: https://github.com/facebookincubator/diffkt/issues/89 Dnnl could probably help us here
        val shapePerIndex = newShape.drop(indices[0].size)
        val elementsPerIndex = shapePerIndex.product
        assert(x.size == indices.size * elementsPerIndex)
        val newData = FloatArray(newShape.product)
        val contigStrides = StridedUtils.contigStrides(newShape)
        for (i in indices.indices) {
            val index = indices[i]
            val srcPos = i * elementsPerIndex
            val destPos = index.foldIndexed(0, { elementIndex, accum, elem -> accum + elem*contigStrides[elementIndex] })
            for (j in 0 until elementsPerIndex) {
                newData[destPos + j] = x.at(srcPos + j)
            }
        }
        return FloatTensor(newShape, newData)
    }

    override fun gamma(alpha: DTensor, randomKey: RandomKey): DTensor {
        return (randomKey as Sha512Random).gamma(alpha as FloatTensor)
    }

    @SType("S: Shape")
    override fun compare(
        left: @SType("S") DTensor,
        right: @SType("S") DTensor,
        comparison: ComparisonKind
    ): @SType("S") DTensor {
        require(left is FloatTensor)
        require(right is FloatTensor)
        return left.zip(right) { l, r -> if (compare(l, r, comparison)) 1f else 0f }
    }

    @SType("S: Shape")
    override fun ifThenElse(
        condition: @SType("S") DTensor,
        whenTrue: @SType("S") DTensor,
        whenFalse: @SType("S") DTensor,
        derivativeId: DerivativeID
    ): @SType("S") DTensor {
        require(condition is FloatTensor)
        require(whenTrue is FloatTensor)
        require(whenFalse is FloatTensor)
        return condition.zip2(whenTrue, whenFalse) { a, b, c -> if (a > 0f) b else c }
    }
}
