/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.resnet

import org.diffkt.model.*
import org.diffkt.*
import kotlin.random.Random

class ResNet private constructor(override val layers: List<Layer<*>>) : Model<ResNet>() {
    override fun withLayers(newLayers: List<Layer<*>>) = ResNet(newLayers)

    override fun hashCode(): Int = combineHash("ResNet", layers)
    override fun equals(other: Any?): Boolean = other is ResNet &&
            other.layers == layers

    companion object {
        operator fun invoke(
            numBlocks: List<Int>,
            numClasses: Int = 10,
            random: Random,
            convWeightInit: (Shape, Random)->FloatTensor = Conv2d.defaultInit): ResNet {

            var inPlanes = 64
            fun makeLayer(planes: Int, numBlocks: Int, firstStride: Int): Sequential {
                val strides = mutableListOf(firstStride)
                (1 until numBlocks).forEach { strides.add(1) }
                val layers = mutableListOf<Layer<*>>()
                for (i in 0 until strides.size) {
                    val stride = strides[i]
                    val block = ResNetBlock(inPlanes, planes, stride, random, convWeightInit)
                    layers.add(block)
                    inPlanes = planes
                }
                return Sequential(layers)
            }

            return ResNet(listOf(
                // Prior to creating the residual blocks, we convolve and normalize the input
                Conv2d(Shape(64, 3, 3, 3), 1, 1, padding = Convolve.Padding2D(1), random = random, weightInit = convWeightInit),
                BatchNormTrainingV1(64),
                ReluLayer,

                // The first set of blocks will not have residual connections because inputPlanes equals planes
                makeLayer(64, numBlocks[0], 1),

                // The subsequent layers will contain residual connections for the first block of the set
                makeLayer(128, numBlocks[1], 2),
                makeLayer(256, numBlocks[2], 2),
                makeLayer(512, numBlocks[3], 2),

                // Pool the output of the residual blocks and add a linear layer to create the proper output size
                AvgPool2d(4, 4),
                Flatten,
                Dense(512, numClasses, random)
            ))
        }
    }
}
