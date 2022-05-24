/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.resnet

import org.diffkt.*
import org.diffkt.model.*
import kotlin.random.Random

/**
 * This is the repeating unit that comprises the ResNet architecture.
 */
class ResNetBlock private constructor(
    private val conv1: Conv2d,
    private val bn1: TrainableLayer<*>,
    private val conv2: Conv2d,
    private val bn2: TrainableLayer<*>,
    private val shortcut: Sequential,
) : TrainableLayerSingleInput<ResNetBlock> {
    override fun hashCode(): Int = combineHash("ResNetBlock", conv1, bn1, conv2, bn2, shortcut)
    override fun equals(other: Any?): Boolean = other is ResNetBlock &&
            other.trainables == trainables

    override val trainables = listOf(conv1, bn1, conv2, bn2, shortcut)

    override fun invoke(input: DTensor): DTensor {
        var out = bn1(conv1(input)).relu()
        out = bn2(conv2(out))
        // The input is convolved to the correct shape and added directly to the output before activation
        out += shortcut(input)
        return out.relu()
    }

    override fun withTrainables(trainables: List<Trainable<*>>): ResNetBlock {
        require(trainables.size == 5)
        return ResNetBlock(
            trainables[0] as Conv2d,
            trainables[1] as TrainableLayer<*>,
            trainables[2] as Conv2d,
            trainables[3] as TrainableLayer<*>,
            trainables[4] as Sequential
        )
    }
    companion object {
        operator fun invoke(inPlanes: Int, planes: Int, strides: Int = 1, random: Random, convWeightInit: (Shape, Random)->FloatTensor): ResNetBlock {
            val expansion = 1
            val conv1 = Conv2d(Shape(planes, 3, 3, inPlanes), strides, strides, padding = Convolve.Padding2D(1), random = random, weightInit = convWeightInit)
            val bn1 = BatchNormTrainingV1(planes)
            val conv2 = Conv2d(Shape(planes, 3, 3, planes), 1, 1, padding = Convolve.Padding2D(1), random = random, weightInit = convWeightInit)
            val bn2 = BatchNormTrainingV1(planes)

            // After the first block set, we will be adding the residual connection which is composed of a convolution and batch norm
            val shortcut =
                if (strides == 1 && inPlanes == expansion * planes)
                    Sequential()
                else {
                    val convS = Conv2d(Shape(planes, 1, 1, inPlanes), strides, strides, paddingStyle = Convolve.PaddingStyle.Valid, random = random, weightInit = convWeightInit)
                    val bnS = BatchNormTrainingV1(expansion * planes)
                    Sequential(convS, bnS)
                }
            return ResNetBlock(conv1, bn1, conv2, bn2, shortcut)
        }
    }
}
