/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

/**
 * Return a slice of the input tensor, which includes only some of the indices at [axis].  Specifically,
 * it includes indices from [start] (inclusive) until [end] (exclusive).
 */
fun DTensor.slice(start: Int, end: Int, axis: Int = 0): DTensor = view(start until end, axis)
fun FloatTensor.slice(start: Int, end: Int, axis: Int = 0): FloatTensor = view(start until end, axis) as FloatTensor
