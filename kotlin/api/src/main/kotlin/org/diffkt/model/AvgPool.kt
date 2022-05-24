/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*
import org.diffkt.Convolve.C_AXIS
import org.diffkt.Convolve.H_AXIS
import org.diffkt.Convolve.N_AXIS
import org.diffkt.Convolve.W_AXIS
import org.diffkt.external.Dnnl

/**
 * Computes the average of the pool (poolHeight x poolWidth) for each pool in x with a stride of (poolHeight, poolWidth).
 * Requires that dim H on x be divisible by poolHeight and dim W on x be divisible by poolWidth.
 *
 * @param x: a tensor of rank 4 and shape NHWC
 *
 * Example:
 * >>> val t = FloatTensor(Shape(1, 4, 4, 1), floatArrayOf(1f, 2f, 3f, ..., 16f)
 * >>> AvgPool.avgPool(t, 2, 2).first
 *
 * FloatTensor(Shape(1, 2, 2, 1), floatArrayOf(3.5f, 5.5f, 11.5f, 13.5f))
 */
fun avgPool(
    x: DTensor,
    poolHeight: Int,
    poolWidth: Int
): DTensor {
    require(x.rank >= 4) { "AvgPool must be called with rank >= 4, was ${x.rank}" }
    return x.operations.avgPool(x, poolHeight, poolWidth)
}

fun avgPoolGrad(
    x: DTensor,
    poolHeight: Int,
    poolWidth: Int,
): DTensor {
    require(x.rank >= 4) { "AvgPool must be called with rank >= 4, was ${x.rank}" }
    return x.operations.avgPoolGrad(x, poolHeight, poolWidth)
}
