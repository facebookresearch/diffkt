/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

/**
 * An optimizer that implements the RMSprop optimization algorithm.
 */
open class RMSpropOptimizer<T : Model<T>>(
    val alpha: Float = 0.005f,
    val beta: Float = 0.9f,
) : Optimizer<T>() {

    protected var nextParameter = 0
    protected val meanSquares = mutableListOf<DTensor>()

    override fun tensorTrainingStep(tensor: DTensor, gradient: DTensor): DTensor {
        require(tensor.shape == gradient.shape)
        val square = gradient.pow(2)
        val meanSquare =
            if (nextParameter >= meanSquares.size) {
                assert(nextParameter == meanSquares.size)
                meanSquares.add(square)
                square
            } else {
                val prevMeanSquare = meanSquares[nextParameter]
                val newMeanSquare = beta * prevMeanSquare + (1 - beta) * square
                meanSquares[nextParameter] = newMeanSquare
                newMeanSquare
            }
        val rms = sqrt(meanSquare)
        nextParameter++
        return tensor - alpha * gradient / rms
    }

    override fun afterFit() {
        nextParameter = 0
    }
}
