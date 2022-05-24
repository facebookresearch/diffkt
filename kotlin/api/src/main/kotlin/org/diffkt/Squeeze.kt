/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * Returns a tensor with the singleton dimension at the specified position of the shape removed.
 *
 * Errors if the shape at the specified position is not 1.
 *
 * Example:
 *   >>> val t = FloatTensor.ones(Shape(3, 1, 4))
 *   >>> t.squeeze(1)
 *   FloatTensor(Shape(3, 4), ...)
 */
fun DTensor.squeeze(axis: Int) : DTensor {
    if (axis < 0 || axis >= this.rank)
        throw IllegalArgumentException("squeeze: axis $axis out of range for rank ${this.rank}")
    if (this.shape[axis] != 1)
        throw IllegalArgumentException("squeeze: can only squeeze singleton dimension, " +
                "tried to squeeze axis $axis of shape ${this.shape}")
    return if (this.rank == 1)
        this.operations.reshapeToScalar(this) // return a scalar
    else
        this.operations.squeeze(this, axis)
}
