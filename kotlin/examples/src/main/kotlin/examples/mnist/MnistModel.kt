/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.mnist

import org.diffkt.*
import org.diffkt.model.*
import kotlin.random.Random

/**
 * Loosely based on LeNet 5 architecture. The differences:
 *   - Input is 28x28, not 32x32
 *   - Use SAME padding instead of VALID padding for conv model.layers, so shape is unchanged through conv model.layers
 *   - Max pool instead of average pool
 *   - ReLu activations after each conv layer
 *   - Softmax instead of Gaussian connections
 */
class MnistModel private constructor(override val layers: List<Layer<*>>): Model<MnistModel>() {
    constructor(random: Random) : this(listOf(
        makeConvLayer(6, 5, 5, 1, random),
        makeMaxPoolLayer(),
        makeConvLayer(16, 5, 5, 6, random),
        makeMaxPoolLayer(),
        Flatten,
        Dense(784, 120, random = random),
        Dense(120, 84, random = random),
        Dense(84, 10, random = random)
    )) { }

    override fun withLayers(newLayers: List<Layer<*>>) = MnistModel(newLayers)

    override fun hashCode(): Int = combineHash("MnistModel", layers)
    override fun equals(other: Any?): Boolean = other is MnistModel &&
            other.layers == layers

    companion object {
        // make a convolution layer. Convolution is not implemented yet.
        private fun makeConvLayer(n: Int, h: Int, w: Int, c: Int, random: Random): Conv2dWithSamePadding =
            Conv2dWithSamePadding(Shape(n, h, w, c), 1, 1, Activation.Relu, random)

        private fun makeMaxPoolLayer() = MaxPool2d(2, 2)
    }
}
