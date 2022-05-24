/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

interface Activation : LayerSingleInput<Activation> {
    object Relu : Activation {
        override fun invoke(input: DTensor) = relu(input)
        override fun hashCode(): Int = combineHash("Relu")
        override fun equals(other: Any?): Boolean = other is Relu
    }

    object Identity : Activation {
        override fun invoke(input: DTensor) = input
        override fun hashCode(): Int = combineHash("Identity")
        override fun equals(other: Any?): Boolean = other is Identity
    }

    object Sigmoid : Activation {
        override fun invoke(input: DTensor) = sigmoid(input)
        override fun hashCode(): Int = combineHash("Sigmoid")
        override fun equals(other: Any?): Boolean = other is Sigmoid
    }

    object Tanh : Activation {
        override fun invoke(input: DTensor) = tanh(input)
        override fun hashCode(): Int = combineHash("Tanh")
        override fun equals(other: Any?): Boolean = other is Tanh
    }
}
