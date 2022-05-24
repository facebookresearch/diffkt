/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.Convolve
import org.diffkt.Shape
import kotlin.random.Random

class Conv2dWithSamePadding(
    filterShape: Shape,
    horizontalStride: Int,
    verticalStride: Int,
    activation: Activation = Activation.Identity,
    random: Random,
): Conv2d(filterShape, horizontalStride, verticalStride, activation, Convolve.PaddingStyle.Same, random)
