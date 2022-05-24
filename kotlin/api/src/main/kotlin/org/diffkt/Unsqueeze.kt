/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * Returns a new tensor with a dimension of size one inserted at the specified position of its shape.
 *
 * Example:
 *   >>> val t = FloatTensor.ones(Shape(3, 4))
 *   >>> t.unsqueeze(1)
 *   FloatTensor(Shape(3, 1, 4), ...)
 */
fun DTensor.unsqueeze(axis: Int) : DTensor {
    if (axis < 0 || axis > this.rank)
        throw IllegalArgumentException("unsqueeze: axis $axis out of range for rank ${this.rank}")
    return this.operations.unsqueeze(this, axis)
}
