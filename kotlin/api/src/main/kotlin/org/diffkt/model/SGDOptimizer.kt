/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

/**
 * Stochastic gradient descent optimizer with optional weight decay regularization and momentum parameters.
 *
 * Default arguments are based on fast.ai's (https://docs.fast.ai/learner#Learner)
 *
 * About using default values:
 * - `SGD()` is "Vanilla SGD", with a default alpha of .001f and no weight decay or momentum.
 * - `SGD(weightDecay = true, `momentum` = true)` enables weight decay and momentum with default values:
 *   `weightDecay`=.01f, `momentum`=.9f. `alpha` remains .001f by default.
 */
class SGDOptimizer<T : TrainableComponent<T>>(
    val initialLearningRate: Float = defaultLearningRate,
    val weightDecay: Float = 0f,
    val momentum: Float = 0f,
) : Optimizer<T>() {
    constructor(initialLearningRate: Float = defaultLearningRate, weightDecay: Boolean, momentum: Boolean): this(
        initialLearningRate = initialLearningRate,
        weightDecay = if (weightDecay) .01f else 0f,
        momentum = if (momentum) .9f else 0f
    )

    private var nextParameter = 0
    private var iteration = 0
    private var learningRate = initialLearningRate
    private val velocities = mutableListOf<DTensor>()

    override fun tensorTrainingStep(tensor: DTensor, gradient: DTensor): DTensor {
        require(tensor.shape == gradient.shape)
        val velocity =
            if (momentum == 0f) {
                gradient
            } else if (nextParameter >= velocities.size) {
                // On the first training round, we don't have any velocity to
                // work with so we use the gradient as the velocity
                assert(iteration == 0)
                assert(nextParameter == velocities.size)
                velocities.add(gradient)
                gradient
            } else {
                val prevVelocity = velocities[nextParameter]
                val newVelocity = momentum * prevVelocity + (1 - momentum) * gradient
                velocities[nextParameter] = newVelocity
                newVelocity
            }

        nextParameter++
        return tensor - learningRate * velocity
    }

    override fun afterFit() {
        nextParameter = 0
        iteration++
        learningRate = initialLearningRate / (1 + weightDecay * iteration)
    }

    companion object {
        private const val defaultLearningRate = .001f
    }
}
