/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.samples

import org.diffkt.FloatScalar
import org.diffkt.reshape
import org.diffkt.tensorOf

class DTensorSamples {

    /**
     * Demonstrate the isScalar property
     */
    fun showIsScalar() {

        val s = FloatScalar(2.0f)
        val indicator = s.isScalar
        println("indicator = ${indicator}")
        // output should be
        // indicator = true
    }

    /**
     * Demonstrate the get property with a single index
     */
    fun showGetSingleIndex() {

        val tensor = tensorOf(1.0f, 2.0f, 3.0f, 4.0f)
        val value = tensor[3]
        println("value = ${value}")
        // output should be
        // value = 4.0f

    }

    /**
     * Demonstrate the get property with multidimensional index
     */
    fun showGetMultiIndex() {

        val tensor = tensorOf(1.0f, 2.0f, 3.0f, 4.0f).reshape(2,2)
        val value = tensor[1,1]
        println("value = ${value}")
        // output should be
        // value = 4.0f
    }

    /**
     * Demonstrate the primal property
     */
    fun showPrimal() {

        val tensor = tensorOf(1.0f, 2.0f, 3.0f, 4.0f).reshape(2,2)
        val primal = tensor.primal
        println("primal = ${primal}")
        // output should be
        // primal = tensorOf(1.0f, 2.0f, 3.0f, 4.0f).reshape(Shape(2, 2))
    }

    /**
     * Demonstrate the rank property
     */
    fun showRank() {

        val tensor = tensorOf(1.0f, 2.0f, 3.0f, 4.0f).reshape(2,2)
        val rank = tensor.rank
        println("rank = ${rank}")
        // output should be
        // rank = 2

    }

    /**
     * Demonstrate the size property
     */
    fun showSize() {
        val tensor = tensorOf(1.0f, 2.0f, 3.0f, 4.0f).reshape(2,2)
        val size = tensor.size
        println("size = ${size}")
        // output should be
        // size = 4
    }

    /**
     * Demonstration of the shape property
     */
    fun showShape() {

        val tensor = tensorOf(1.0f, 2.0f, 3.0f, 4.0f).reshape(2,2)
        val shape = tensor.shape
        println("shape = ${shape}")
        // output should be
        // shape = Shape(2, 2)
    }
}

fun main() {

    val t = DTensorSamples()
    t.showIsScalar()
    t.showGetSingleIndex()
    t.showGetMultiIndex()
    t.showPrimal()
    t.showRank()
    t.showSize()
    t.showShape()

}