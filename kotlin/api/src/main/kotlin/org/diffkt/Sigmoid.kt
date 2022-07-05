/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import kotlin.math.exp
import shapeTyping.annotations.SType
import org.diffkt.adOptimize.ToUnboxedFunction

/**
 * Compute the sigmoid for a single floating-point value.
 */
internal fun sigmoidElem(x: Float): Float {
    return if (x > 0) {
        val intermediate = exp(-x)
        1f / (1f + intermediate)
    } else {
        val intermediate = exp(x)
        intermediate / (1 + intermediate)
    }
}

@ToUnboxedFunction("org.diffkt.sigmoidElem")
fun sigmoid(x: DScalar): DScalar {
    return x.operations.sigmoid(x) as DScalar
}

/**
 * Returns [x] with the sigmoid activation unit applied (1 / (1 + exp(-x))
 */
@SType("S: Shape")
fun sigmoid(x: @SType("S") DTensor): @SType("S") DTensor {
    return x.operations.sigmoid(x)
}
