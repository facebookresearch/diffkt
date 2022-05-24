/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

abstract class Model<T : Model<T>> : TrainableComponent<T> {
    abstract val layers: List<Layer<*>>
    override val trainables: List<Trainable<*>>
        get() = layers.filterIsInstance<Trainable<*>>()
    open fun predict(data: DTensor): DTensor {
        return layers.fold(data, { accum, layer -> layer.invoke(accum) })
    }
    abstract fun withLayers(newLayers: List<Layer<*>>): T

    override fun withTrainables(trainables: List<Trainable<*>>): T {
        val trainableArray = trainables.toTypedArray()
        var nextTrainable = 0
        val newLayers = layers.map { if (it is Trainable<*>) trainableArray[nextTrainable++] as Layer<*> else it }
        assert(nextTrainable == trainableArray.size)
        return withLayers(newLayers)
    }

    override fun wrap(wrapper: Wrapper): T {
        return withLayers(layers.map { wrapper.wrap(it) })
    }

    override fun cpu(): T {
        return withLayers(layers.map { it.cpu()} )
    }

    override fun gpu(): T {
        return withLayers(layers.map { it.gpu() })
    }

    abstract override fun hashCode(): Int
    abstract override fun equals(other: Any?): Boolean
}
