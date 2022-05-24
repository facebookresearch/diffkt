/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import shapeTyping.annotations.SType

/** Create a tensor of the same shape, with all elements flipped across the specified axes */
@SType("S: Shape")
fun @SType("S") DTensor.flip(axes: IntArray): @SType("S") DTensor {
    return this.operations.flip(this, axes)
}

/** Create a tensor of the same shape, with all elements flipped across the specified axes */
@JvmName("ExtensionVarargFlip")
@SType("S: Shape")
fun @SType("S") DTensor.flip(vararg axes: Int): @SType("S") DTensor = this.flip(axes)
