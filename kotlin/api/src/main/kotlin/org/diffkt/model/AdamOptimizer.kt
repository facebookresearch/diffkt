/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.DTensor

/**
 * THIS IS A PLACEHOLDER!
 */
class AdamOptimizer<T : Model<T>> : Optimizer<T>() {
    override fun tensorTrainingStep(tensor: DTensor, gradient: DTensor): DTensor {
        TODO("Not yet implemented")
    }
    override fun afterFit() {
        TODO("Not yet implemented")
    }
}
