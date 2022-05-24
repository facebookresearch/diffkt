/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.DTensor
import org.diffkt.combineHash

class MaxPool2d(
    val poolHeight: Int,
    val poolWidth: Int
) : LayerSingleInput<MaxPool2d> {
    override fun invoke(input: DTensor): DTensor {
        return maxPool(input, poolHeight, poolWidth)
    }

    override fun hashCode(): Int = combineHash("MaxPool2d", poolHeight, poolWidth)
    override fun equals(other: Any?): Boolean = other is MaxPool2d &&
            other.poolHeight == poolHeight &&
            other.poolWidth == poolWidth
}
