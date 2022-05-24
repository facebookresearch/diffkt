/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

/**
 * simple optimizer that just uses a fixed learning rate.
 */
class FixedLearningRateOptimizer<T : Model<T>>(val alpha: DScalar) : Optimizer<T>() {
    constructor(alpha: Float) : this(FloatScalar(alpha))
    override fun tensorTrainingStep(tensor: DTensor, gradient: DTensor): DTensor {
        require(tensor.shape == gradient.shape)
        return tensor - alpha * gradient
    }
}
