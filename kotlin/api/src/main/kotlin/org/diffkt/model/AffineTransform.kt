/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

/**
 * An affine transform.  Multiplies by one tensor and then adds another.
 * Like a [Dense] layer, except that where a dense layer performs a
 * matmul, this one performs an element-wise multiplication.
 */
class AffineTransform(val m: TrainableTensor, val b: TrainableTensor): TrainableLayerSingleInput<AffineTransform> {
    override fun wrap(wrapper: Wrapper): AffineTransform {
        return AffineTransform(wrapper.wrap(m), wrapper.wrap(b))
    }

    override fun invoke(input: DTensor): DTensor {
        return m.tensor * input + b.tensor
    }

    override val trainables: List<Trainable<*>>
        get() = listOf(m, b)

    override fun withTrainables(trainables: List<Trainable<*>>): AffineTransform {
        assert(trainables.size == 2 && trainables.all { it is TrainableTensor })
        val values = trainables.toTypedArray()
        return AffineTransform(values[0] as TrainableTensor, values[1] as TrainableTensor)
    }

    override fun hashCode(): Int = combineHash("AffineTransform", m, b)
    override fun equals(other: Any?): Boolean = other is AffineTransform &&
            other.m == m &&
            other.b == b
}