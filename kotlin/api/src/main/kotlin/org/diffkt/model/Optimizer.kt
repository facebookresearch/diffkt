/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.DTensor

abstract class Optimizer<T : Trainable<T>> {
    /**
     * Train a trainable using the given tangent.
     */
    fun train(component: T, tangent: Trainable.Tangent): T {
        val trainedComponent: T = component.trainingStep(this, tangent)
        this.afterFit()
        return trainedComponent
    }

    /**
     * Train an element of the model.  For the model itself, use [train].
     */
    abstract fun tensorTrainingStep(tensor: DTensor, gradient: DTensor): DTensor

    protected open fun afterFit() {}
}
