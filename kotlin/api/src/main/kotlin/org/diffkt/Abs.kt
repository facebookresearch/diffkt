/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import shapeTyping.annotations.SType

fun abs(x: DScalar): DScalar {
    val t = -x
    return ifThenElse(t, t, x)
}

@SType("S: Shape")
fun abs(x: @SType("S") DTensor): @SType("S") DTensor {
    val t = -x
    return ifThenElse(t, t, x)
}
