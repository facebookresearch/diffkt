/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

/**
 * Fan mode informs weight initializers of the dimension for the weight matrix
 * where Fan In corresponds to the size of incoming data and Fan Out corresponds to
 * the size of outgoing data.
 */
sealed class FanMode {
    fun fanFactor(shape: Shape): Int {
        val dims = shape.rank
        require(dims >= 2) { "Input tensor has dimensions $shape, fan-in/out cannot be computed on" +
                " fewer than 2 dimensions" }
        return fanSize(shape) * shape.drop(2).product
    }

    abstract fun fanSize(shape: Shape): Int
}

class FanIn : FanMode() {
    override fun fanSize(shape: Shape) = shape[1]
}

class FanOut : FanMode() {
    override fun fanSize(shape: Shape) = shape.first
}
