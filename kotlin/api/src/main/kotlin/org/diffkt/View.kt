/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

fun DTensor.view(indices: IntArray): DTensor {
    require(indices.size <= shape.rank)
    require(indices.indices.all { indices[it] >= 0 && indices[it] < shape[it] })
    return this.operations.view1(this, indices)
}

fun DTensor.view(index: Int, axis: Int): DTensor {
    require(axis >= 0 && axis < shape.rank)
    require(index >= 0 && index < shape[axis])
    return this.operations.view2(this, index, axis)
}

fun DTensor.view(index: IntRange, axis: Int): DTensor {
    require(axis >= 0 && axis < shape.rank)
    require(index.start >= 0 && index.endInclusive < shape[axis])
    require(index.step == 1) // nontrivial steps not yet supported
    require(index.start <= index.endInclusive)
    return this.operations.view3(this, index, axis)
}
