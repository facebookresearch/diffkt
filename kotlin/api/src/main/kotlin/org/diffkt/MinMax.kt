/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

fun FloatTensor.max(axes: IntArray = allAxes, keepDims: Boolean = false): FloatTensor {
    return this.reduce({ x, y -> kotlin.math.max(x, y) }, axes, keepDims)
}

fun FloatTensor.min(axes: IntArray = allAxes, keepDims: Boolean = false): FloatTensor {
    return this.reduce({ x, y -> kotlin.math.min(x, y) }, axes, keepDims)
}
