/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils

import org.diffkt.FloatTensor

fun Pair<FloatTensor, FloatTensor>.mapIndexed(f: (Int, Float, Float) -> Pair<Float, Float>): Pair<FloatTensor, FloatTensor> {
    val (x, y) = this
    require(x.shape == y.shape)
    val valuesPair = List(x.shape.product()) { f(it, x.at(it), y.at(it)) }.unzip()
    return Pair(
        FloatTensor(x.shape, valuesPair.first.toFloatArray()),
        FloatTensor(y.shape, valuesPair.second.toFloatArray())
    )
}
