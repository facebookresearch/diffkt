/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.samples

import org.diffkt.DScalar
import org.diffkt.FloatScalar

class DScalarSamples {

    fun showPrimal() {

        var s : DScalar
        s = FloatScalar(1.0f)
        val primal = s.primal
        println("primal = ${primal}")
        // output should be
        // primal = 1.0F
    }

    fun showRank() {

        var s : DScalar
        s = FloatScalar(1.0f)
        val rank = s.rank
        println("rank = ${rank}")
        // output should be
        // rank = 0


    }

    fun showShape() {

        var s :DScalar
        s = FloatScalar(1.0f)
        val shape = s.shape
        println("shape = ${shape}")
        // output should be
        // shape = Shape()

    }

}


fun main() {

    val s = DScalarSamples()
    s.showPrimal()
    s.showRank()
    s.showShape()
}