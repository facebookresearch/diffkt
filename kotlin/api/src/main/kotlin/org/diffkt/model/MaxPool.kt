/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*
import org.diffkt.external.Dnnl

/**
 * Returns the max of the pool (poolHeight x poolWidth) for each pool in x with a stride of (poolHeight, poolWidth).
 * Requires that dim H on x be divisible by poolHeight and dim W on x be divisible by poolWidth
 *
 * @param x: a tensor of rank 4 and shape NHWC
 *
 * Example:
 * >>> val t = FloatTensor(Shape(1, 4, 4, 1), floatArrayOf(1f, 2f, 3f, ..., 16f)
 * >>> MaxPool.maxPool(t, 2, 2)
 *
 * FloatTensor(Shape(1, 2, 2, 1), floatArrayOf(6f, 8f, 14f, 16f))
 */
fun maxPool(
    x: DTensor,
    poolHeight: Int,
    poolWidth: Int
): DTensor {
    require(x.rank == 4) { "MaxPool must be called with rank 4, was ${x.rank}" }
    return x.operations.maxPoolWithIndices(x, poolHeight, poolWidth, withIndices = false).first
}

internal fun maxPoolWithIndicesDnnl(
    x: StridedFloatTensor,
    poolHeight: Int,
    poolWidth: Int,
    withIndices: Boolean
): Pair<FloatTensor, ByteArray?> {
    require(x.rank == 4) { "MaxPool must be called with rank 4, was ${x.rank}" }
    val numItems = x.shape[Convolve.N_AXIS]
    val inHeight = x.shape[Convolve.H_AXIS]
    val inWidth = x.shape[Convolve.W_AXIS]
    val numChannels = x.shape[Convolve.C_AXIS]
    require(inHeight % poolHeight == 0) {
        "input height ($inHeight) must be divisible by pool height ($poolHeight)" }
    require(inWidth % poolWidth == 0) {
        "input width ($inWidth) must be divisible by pool width ($poolWidth)" }
    val outHeight = inHeight / poolHeight
    val outWidth = inWidth / poolWidth
    val outShape = Shape(numItems, outHeight, outWidth, numChannels)
    val values = FloatArray(outShape.product)
    val maxIndices = ByteArray(outShape.product)
    Dnnl.maxPool(
        // result
        outShape.dims,
        values,
        // max indices for later gradient calculation
        maxIndices,
        // input
        x.shape.dims,
        x.data,
        // pool height and width
        poolHeight,
        poolWidth
    )
    return Pair(FloatTensor(outShape, values), if (withIndices) maxIndices else null)
}
