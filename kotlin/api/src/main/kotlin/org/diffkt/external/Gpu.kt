/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.external

object Gpu : ExternalLib {
    private const val DYLIB_NAME = "libgpuops_jni"
    private var _isLoaded = false

    override val isLoaded get() = _isLoaded

    init {
        try {
            loadLib(DYLIB_NAME)
            _isLoaded = true
        } catch (e: Exception) { }
    }

    // --- Tensor utils ---
    external fun deleteHandle(handle: Long)
    external fun getShape(handle: Long): IntArray
    external fun getFloatData(handle: Long): FloatArray
    external fun putFloatTensor(shape: IntArray, data: FloatArray): Long

    external fun zeros(shape: IntArray): Long

    // --- Misc utils ---
    external fun getAllocatedBytes(): Long

    // --- Ops ---
    external fun add(lhs: Long, rhs: Long): LongArray
    external fun addGradLhs(seed: Long, lhs: Long, forwardRes: Long): Long
    external fun addGradRhs(seed: Long, rhs: Long, forwardRes: Long): Long

    external fun avgPool(x: Long, poolHeight: Int, poolWidth: Int): LongArray
    external fun avgPoolGrad(seed: Long, x: Long, forwardRes: Long): Long

    external fun batchNorm2d(input: Long, scaleShift: Long, runningMean: Long, runningVariance: Long, momentum: Float): LongArray
    external fun batchNorm2dGradInput(seed: Long, input: Long, forwardRes: Long): Long
    external fun batchNorm2dGradScaleShift(seed: Long, scaleShift: Long, forwardRes: Long): Long

    external fun broadcastTo(seed: Long, newShape: IntArray): LongArray
    external fun broadcastToGrad(seed: Long, forwardArg: Long, forwardRes: Long): Long

    external fun conv2d(images: Long, filters: Long, strides: IntArray, padding: IntArray): LongArray
    external fun conv2dGradImages(seed: Long, images: Long, forwardRes: Long): Long
    external fun conv2dGradFilters(seed: Long, filters: Long, forwardRes: Long): Long

    external fun div(lhs: Long, rhs: Long): Long

    external fun logSoftmax(x: Long, axis: Int): LongArray
    external fun logSoftmaxGrad(seed: Long, x: Long, forwardRes: Long): Long

    external fun matmul(lhs: Long, rhs: Long): LongArray
    external fun matmulGradLhs(seed: Long, lhs: Long, forwardRes: Long): Long
    external fun matmulGradRhs(seed: Long, rhs: Long, forwardRes: Long): Long

    external fun maxPool(x: Long, poolHeight: Int, poolWidth: Int): LongArray
    external fun maxPoolGrad(seed: Long, x: Long, forwardRes: Long): Long

    external fun nllLoss(x: Long, labels: Long): LongArray
    external fun nllLossGradX(seed: Long, x: Long, forwardRes: Long): Long
    external fun nllLossGradLabels(seed: Long, labels: Long, forwardRes: Long): Long

    external fun relu(handle: Long): LongArray
    external fun reluGrad(seed: Long, forwardArg: Long, forwardRes: Long): Long

    external fun reshape(handle: Long, newShape: IntArray): LongArray
    external fun reshapeGrad(seed: Long, forwardArg: Long, forwardRes: Long): Long

    external fun sub(lhs: Long, rhs: Long): Long
    external fun isub(lhs: Long, rhs: Long): Long

    external fun sum(handle: Long, axes: IntArray, keepDims: Boolean): LongArray
    external fun sumGrad(seed: Long, forwardArg: Long, forwardRes: Long): Long

    external fun times(lhs: Long, rhs: Long): Long
}
