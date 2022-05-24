/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.iris

import org.diffkt.combineHash
import org.diffkt.data.loaders.iris.IrisDataLoader
import org.diffkt.model.*
import kotlin.random.Random

class IrisModel(private val d1: Dense, private val d2: Dense): Model<IrisModel>() {
    constructor(random: Random) : this(
        d1 = Dense(
            IrisDataLoader.NUM_FEATURES,
            IrisDataLoader.MIDDLE_DATA,
            random,
            activation = Activation.Relu,
            weightInit = Initializer.uniform(),
            biasInit = Initializer.uniform()),
        d2 = Dense(
            IrisDataLoader.MIDDLE_DATA,
            IrisDataLoader.NUM_SPECIES,
            random,
            activation = Activation.Identity,
            weightInit = Initializer.uniform(),
            biasInit = Initializer.uniform()))

    override val layers: List<Layer<*>> = listOf(d1, d2)

    override fun withLayers(newLayers: List<Layer<*>>): IrisModel {
        return IrisModel(newLayers[0] as Dense, newLayers[1] as Dense)
    }

    override fun hashCode(): Int = combineHash("IrisModel", d1, d2)
    override fun equals(other: Any?): Boolean = other is IrisModel && other.d1 == d1 && other.d2 == d2
}
