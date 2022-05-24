/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.DTensor
import org.diffkt.combineHash

/**
 * Average pool 2d
 *
 * Downsamples the input, returning the average for each pool/window.
 */
class AvgPool2d(
    val poolHeight: Int,
    val poolWidth: Int
) : LayerSingleInput<AvgPool2d> {
    override fun invoke(input: DTensor): DTensor {
        return avgPool(input, poolHeight, poolWidth)
    }
    override fun hashCode(): Int = combineHash("AvgPool2d", poolHeight, poolWidth)
    override fun equals(other: Any?): Boolean = other is AvgPool2d &&
            other.poolHeight == poolHeight &&
            other.poolWidth == poolWidth
}
