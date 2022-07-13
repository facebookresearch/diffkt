/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import shapeTyping.annotations.SType
import org.diffkt.adOptimize.ScalarRoot

/**
 * A differentiable scalar (float).
 * We represent a number [DScalar] as either
 * - a [FloatScalar], which is a wrapper around a float, or
 * - a [ForwardScalar] for forward differentiation with a [DScalar] primal value, and a [DTensor] tangent, or
 * - a [ReverseScalar] for reverse mode differentiation.
 */
@ScalarRoot
interface DScalar : @SType("[]") DTensor {

    /**
     * primal points to the actual tensor
     * @sample org.diffkt.samples.DScalarSamples.showPrimal
     * */
    override val primal: DScalar

    /**
     * rank is the number of dimension of a tensor.
     * scalars always have a rank of 0
     * @sample org.diffkt.samples.DScalarSamples.showRank
     */
    override val rank: Int get() = 0

    /**
     * shape indicates the number of dimensions and the length of the dimensions.
     * Since scalars have rank 0, shape always returns Shape()
     * @sample org.diffkt.samples.DScalarSamples.showShape
     */
    override val shape: @SType("[]") Shape get() = Shape()
}