/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.DTensor
import org.diffkt.combineHash

class Sequential(val layers: List<Layer<*>>) : TrainableLayerSingleInput<Sequential> {

    constructor(vararg layers: Layer<*>) : this(listOf(*layers))

    override operator fun invoke(input: DTensor): DTensor {
        return layers.fold(input) { x, op -> op.invoke(x) }
    }

    override val trainables: List<Trainable<*>>
        get() = layers.filterIsInstance<Trainable<*>>()

    override fun withTrainables(trainables: List<Trainable<*>>): Sequential {
        var nextTrainable = 0
        val newLayers = layers.map { if (it is Trainable<*>) trainables[nextTrainable++] as Layer<*> else it }
        assert(nextTrainable == trainables.size)
        return Sequential(*newLayers.toTypedArray())
    }

    override fun hashCode(): Int = combineHash("Sequential", combineHash(*layers.toTypedArray()))
    override fun equals(other: Any?): Boolean = other is Sequential &&
            other.layers == layers
}
