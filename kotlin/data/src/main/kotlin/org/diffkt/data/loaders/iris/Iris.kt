/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.data.loaders.iris

class Iris(
    sepalLength: Float,
    sepalWidth: Float,
    petalLength: Float,
    petalWidth: Float
) {
    val features = mutableListOf(sepalLength, sepalWidth, petalLength, petalWidth)
}
