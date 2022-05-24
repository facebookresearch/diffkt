/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * Return a tensor made up of slices taken from the input tensor at the given indices
 * along the given axis.
 *
 * @param indices indices to gather slices from
 * @param axis axis long which to slice x
 * @return result tensor
 */

fun DTensor.gather(indices: List<Int>, axis: Int, paddingIndex: Int = -1): DTensor =
    this.operations.gather(this, indices, axis, paddingIndex)

fun FloatTensor.gather(indices: List<Int>, axis: Int, paddingIndex: Int = -1): FloatTensor =
    (this as DTensor).operations.gather(this, indices, axis, paddingIndex) as FloatTensor