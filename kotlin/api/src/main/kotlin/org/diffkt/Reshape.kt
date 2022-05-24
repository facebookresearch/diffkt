/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

@file:Suppress("UNREDUCED_STYPE_ERROR") // shapeTyping #72: affects extension functions referencing `this` and shaped properties
package org.diffkt

fun DTensor.reshape(vararg newShape: Int): DTensor = reshape(Shape(newShape))

fun DTensor.reshape(newShape: Shape): DTensor {
    // Check that new shape is valid.
    val oldShape = this.shape
    if (newShape == oldShape) return this
    require(oldShape.product() == newShape.product())
    return if (newShape.isScalar)
        this.operations.reshapeToScalar(this)
    else
        this.operations.reshape(this, newShape)
}
