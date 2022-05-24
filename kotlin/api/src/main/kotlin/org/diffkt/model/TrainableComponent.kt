/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*
import java.nio.ByteBuffer
import java.util.*

interface TrainableComponent<T : TrainableComponent<T>> : Trainable<T> {
    val trainables: List<Trainable<*>>

    override fun trainingStep(optim: Optimizer<*>, tangent: Trainable.Tangent): T {
        require(tangent is Tangent)
        val newTrainables =
            trainables.zip(tangent.grads) { trainable, grad -> trainable.trainingStep(optim, grad) }
        return withTrainables(newTrainables)
    }

    override fun extractTangent(output: DTensor, extractor: (DTensor, DTensor) -> DTensor): Tangent {
        val grads = trainables.map { it.extractTangent(output, extractor) }
        return Tangent(grads)
    }

    override fun store(into: ByteBuffer): ByteBuffer {
        return trainables.fold(into) { r, t ->
            t.store(r)
        }
    }

    override fun load(from: ByteBuffer): T {
        val newTrainables =
            trainables.map { trainable -> trainable.load(from) }
        return withTrainables(newTrainables)
    }

    fun withTrainables(trainables: List<Trainable<*>>): T

    override fun wrap(wrapper: Wrapper): T = withTrainables(trainables.map { wrapper.wrap(it) })

    override fun cpu(): T = withTrainables(trainables.map { it.cpu() })
    override fun gpu(): T = withTrainables(trainables.map { it.gpu() })

    companion object {
        data class Tangent(val grads: Collection<Trainable.Tangent>) : Trainable.Tangent
    }
}