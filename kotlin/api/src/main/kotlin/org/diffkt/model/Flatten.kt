/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

/**
 * Flattens the input. Does not affect batch size.
 *
 * Shape transform: (N, *) -> (N, <product of *>)
 */
object Flatten : LayerSingleInput<Flatten> {
    override fun invoke(input: DTensor): DTensor {
        return input.flatten(startDim = 1)
    }
}
