/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*
import kotlin.math.sqrt
import kotlin.random.Random

open class Conv2d(
    val filterShape: Shape,
    val horizontalStride: Int,
    val verticalStride: Int,
    val activation: Activation = defaultActivation,
    val paddingStyle: Convolve.PaddingStyle,
    private val trainableFilter: TrainableTensor,
) : TrainableLayerSingleInput<Conv2d> {
    constructor(
        filterShape: Shape,
        horizontalStride: Int,
        verticalStride: Int,
        activation: Activation = defaultActivation,
        padding: Convolve.Padding2D,
        random: Random,
        weightInit: (Shape, Random)-> FloatTensor = defaultInit,
    ) : this(
        filterShape, horizontalStride, verticalStride, activation,
        Convolve.PaddingStyle.Explicit(padding.top, padding.bottom,
            padding.left, padding.right), TrainableTensor(weightInit(filterShape, random)))
    constructor(
        filterShape: Shape,
        horizontalStride: Int,
        verticalStride: Int,
        activation: Activation = defaultActivation,
        paddingStyle: Convolve.PaddingStyle,
        random: Random,
        weightInit: (Shape, Random)->FloatTensor = defaultInit,
    ) : this(
        filterShape, horizontalStride, verticalStride, activation,
        paddingStyle, TrainableTensor(weightInit(filterShape, random)))

    override val trainables: List<Trainable<*>> = listOf(trainableFilter)

    override fun withTrainables(trainables: List<Trainable<*>>): Conv2d {
        require(trainables.size == 1)
        val newTT = trainables[0] as TrainableTensor
        return Conv2d(filterShape, horizontalStride, verticalStride, activation, paddingStyle, newTT)
    }

    override fun invoke(input: DTensor): DTensor {
        return activation(
            conv2d(input, trainableFilter.tensor, horizontalStride, verticalStride,
            paddingStyle = paddingStyle)
        )
    }

    override fun hashCode(): Int =
        combineHash("Conv2d", filterShape, horizontalStride, verticalStride, activation, paddingStyle, trainableFilter)
    override fun equals(other: Any?): Boolean = other is Conv2d &&
            other.filterShape == filterShape &&
            other.horizontalStride == horizontalStride &&
            other.verticalStride == verticalStride &&
            other.activation == activation &&
            other.paddingStyle == paddingStyle &&
            other.trainableFilter == trainableFilter

    companion object {
        // Default values for constructors
        private val defaultActivation = Activation.Identity
        val defaultInit = Initializer.kaimingUniform(FanIn(), Initializer.kaimingUniform.LeakyReluTransformGainFactor(sqrt(5f)))
    }
}
