/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.external

import org.diffkt.FloatTensor
import org.diffkt.Shape
import org.diffkt.StridedFloatTensor

object Dnnl: ExternalLib {
    private const val DYLIB_NAME = "libdnnlops_jni"
    private var _isLoaded = false

    override val isLoaded get() = _isLoaded

    init {
        try {
            loadLib(DYLIB_NAME)
            _isLoaded = true
        } catch (e: Exception) { }
    }


    fun add(left: StridedFloatTensor, right: StridedFloatTensor): StridedFloatTensor {
        require(left.shape == right.shape) { "Add requires matching tensor shapes" }
        return StridedFloatTensor.contiguous(left.shape) {
            add(left.shape.dims, left.strides, right.strides, left.offset, right.offset, it, left.data, right.data)
        }
    }

    fun sub(left: StridedFloatTensor, right: StridedFloatTensor): StridedFloatTensor {
        require(left.shape == right.shape) { "Sub requires matching tensor shapes" }
        return StridedFloatTensor.contiguous(left.shape) {
            sub(left.shape.dims, left.strides, right.strides, left.offset, right.offset, it, left.data, right.data)
        }
    }

    fun matmul(left: StridedFloatTensor, right: StridedFloatTensor, a: Shape, b: Shape, d: Shape): StridedFloatTensor {
        val newShape = a + b + d
        val res = FloatArray(newShape.product)
        matmul(left.shape.dims, left.strides, left.offset, right.shape.dims, right.strides, right.offset, res, left.data, right.data)
        return StridedFloatTensor(newShape, res)
    }

    fun mulScalar(x: FloatTensor, alpha: Float): FloatTensor {
        val xn = x.normalize()
        return StridedFloatTensor.contiguous(x.shape) {
            // mulScalar(x.shape.dims, it, x.normalize().data, alpha)
            linear(xn.shape.dims, xn.strides, xn.offset,  it, xn.data, alpha, 0f)
        }
    }

    /**
     * Convenience wrapper for DNNL batchnorm grad.
     *
     * @return Pair(input grad, scale-and-shift grad)
     */
    fun batchNormGrad(
            seed: FloatTensor,
            input: FloatTensor,
            scaleShift: FloatTensor,
            mean: FloatTensor,
            variance: FloatTensor
    ): Pair<FloatTensor, FloatTensor> {
        require(input.rank == 4 && input.shape == seed.shape) {
            "input and seed must be rank 4 and have the same shape"
        }
        val C = input.shape[3]
        require(mean.shape == Shape(C) && variance.shape == mean.shape) { "mean and variance must have Shape($C)" }
        require(scaleShift.shape == Shape(2, C)) { "scaleShift must have shape ${Shape(2, C)}" }

        val inputGrad = StridedFloatTensor.contigZeros(input.shape)
        val scaleShiftGrad = StridedFloatTensor.contigZeros(scaleShift.shape)

        batchNormGrad(inputGrad.shape.dims, inputGrad.data, scaleShiftGrad.data,
                seed.normalize().data, input.normalize().data, scaleShift.normalize().data, mean.normalize().data,
                variance.normalize().data)
        return Pair(inputGrad, scaleShiftGrad)
    }

    // --- External functions ---
    private external fun add(
            shape: IntArray,
            lhsStrides: IntArray,
            rhsStrides: IntArray,
            lhsOffset: Int,
            rhsOffset: Int,
            result: FloatArray,
            lhs: FloatArray,
            rhs: FloatArray
    )

    external fun batchNorm(
            resultShape: IntArray,
            result: FloatArray,
            mean: FloatArray,
            variance: FloatArray,
            input: FloatArray,
            scaleShift: FloatArray
    )

    private external fun batchNormGrad(
            resultShape: IntArray,
            inputGrad: FloatArray,
            scaleShiftGrad: FloatArray,
            seed: FloatArray,
            input: FloatArray,
            scaleShift: FloatArray,
            mean: FloatArray,
            variance: FloatArray
    )

    external fun conv2d(
            resultShape: IntArray,
            result: FloatArray,
            inputShape: IntArray,
            input: FloatArray,
            filtersShape: IntArray,
            filters: FloatArray,
            hstride: Int,
            vstride: Int,
            paddingLeft: Int,
            paddingRight: Int,
            paddingTop: Int,
            paddingBottom: Int
    )

    external fun conv2dGradImage(
            resultShape: IntArray,
            result: FloatArray,
            seedShape: IntArray,
            seed: FloatArray,
            filtersShape: IntArray,
            filters: FloatArray,
            hstride: Int,
            vstride: Int,
            paddingLeft: Int,
            paddingRight: Int,
            paddingTop: Int,
            paddingBottom: Int
    )

    external fun conv2dGradFilter(
            resultShape: IntArray,
            result: FloatArray,
            seedShape: IntArray,
            seed: FloatArray,
            imagesShape: IntArray,
            images: FloatArray,
            hstride: Int,
            vstride: Int,
            paddingLeft: Int,
            paddingRight: Int,
            paddingTop: Int,
            paddingBottom: Int
    )

    external fun linear(
            shape: IntArray,
            strides: IntArray,
            offset: Int,
            res: FloatArray,
            input: FloatArray,
            scale: Float,
            shift: Float
    )

    external fun logSoftmax(
            shape: IntArray,
            input: FloatArray,
            res: FloatArray,
            axis: Int
    )

    /** Given the result of the forward op and the seed, returns the grad */
    external fun logSoftmaxGrad(
            shape: IntArray,
            grad: FloatArray,
            seed: FloatArray,
            fwdRes: FloatArray,
            axis: Int
    )

    external fun maxPool(
            resultShape: IntArray,
            result: FloatArray,
            workspace: ByteArray,
            imagesShape: IntArray,
            images: FloatArray,
            poolHeight: Int,
            poolWidth: Int
    )

    external fun maxPoolGrad(
            resultShape: IntArray,
            result: FloatArray,
            workspace: ByteArray,
            seedShape: IntArray,
            seed: FloatArray,
            poolHeight: Int,
            poolWidth: Int
    )

    private external fun mulScalar(
            shape: IntArray,
            result: FloatArray,
            lhs: FloatArray,
            rhs: Float
    )

    external fun avgPool(
            resultShape: IntArray,
            result: FloatArray,
            imagesShape: IntArray,
            images: FloatArray,
            poolHeight: Int,
            poolWidth: Int
    )

    external fun avgPoolGrad(
            resultShape: IntArray,
            result: FloatArray,
            seedShape: IntArray,
            seed: FloatArray,
            poolHeight: Int,
            poolWidth: Int
    )

    external fun reduceSum(
            resultShape: IntArray,
            result: FloatArray,
            inputShape: IntArray,
            input: FloatArray
    )

    external fun relu(
            shape: IntArray,
            result: FloatArray,
            input: FloatArray
    )

    external fun reluGrad(
            shape: IntArray,
            result: FloatArray,
            seed: FloatArray,
            input: FloatArray
    )

    private external fun sub(
            shape: IntArray,
            lhsStrides: IntArray,
            rhsStrides: IntArray,
            lhsOffset: Int,
            rhsOffset: Int,
            result: FloatArray,
            lhs: FloatArray,
            rhs: FloatArray
    )

    external fun matmul(
            lhsShape: IntArray,
            lhsStrides: IntArray,
            lhsOffset: Int,
            rhsShape: IntArray,
            rhsStrides: IntArray,
            rhsOffset: Int,
            result: FloatArray,
            lhs: FloatArray,
            rhs: FloatArray,
    )
}
