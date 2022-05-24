/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import shapeTyping.annotations.SType

// Note: expand does not currently handle wildcard (-1) axis values
@SType("S: Shape")
fun DTensor.expand(newShape: @SType("S") Shape): @SType("S") DTensor {
    require(rank == newShape.rank) { "expand: new shape $newShape and current shape $shape have different ranks." }
    return this.operations.expand(this, newShape)
}
