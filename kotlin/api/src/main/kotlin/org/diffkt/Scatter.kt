/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

fun DTensor.scatter(indices: List<Int>, axis: Int, newShape: Shape, paddingIndex: Int = -1): DTensor =
    this.operations.scatter(this, indices, axis, newShape, paddingIndex)


fun FloatTensor.scatter(indices: List<Int>, axis: Int, newShape: Shape, paddingIndex: Int = -1): FloatTensor =
    (this as DTensor).operations.scatter(this, indices, axis, newShape, paddingIndex) as FloatTensor