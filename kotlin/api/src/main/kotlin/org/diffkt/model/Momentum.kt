/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.DTensor
import org.diffkt.plus
import org.diffkt.times

/**
 * Returns the current (this) tensor updated by the new tensor (new) scaled by momentum
 */
fun DTensor.momentumUpdated(new: DTensor, momentum: Float): DTensor {
    return (1 - momentum) * this + momentum * new
}

/**
 * Returns the current (this) value updated by the new value (new) scaled by momentum
 */
fun Float.momentumUpdated(new: Float, momentum: Float): Float {
    return (1 - momentum) * this + momentum * new
}
