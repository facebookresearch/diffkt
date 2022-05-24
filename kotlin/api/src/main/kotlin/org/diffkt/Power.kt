/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import shapeTyping.annotations.SType

// Tensor powers

// a^b == e^(b ln a)
fun DTensor.pow(x: DScalar): DTensor = exp(x * ln(this))

// a^b == e^(b ln a)
fun DScalar.pow(x: DScalar): DScalar = exp(x * ln(this))

fun DScalar.pow(x: Float): DScalar = (this as DTensor).pow(x) as DScalar

fun DScalar.pow(x: Int): DScalar = (this as DTensor).pow(x.toFloat()) as DScalar

fun DTensor.pow(x: Int): DTensor = this.pow(x.toFloat())

@SType("S: Shape")
fun @SType("S") DTensor.pow(exponent: @SType("S") DTensor): @SType("S") DTensor = exp(exponent * ln(this))

@SType("S: Shape")
fun @SType("S") DTensor.pow(exponent: Float): @SType("S") DTensor {
    // optimized base case to prevent recursing into unnecessary computation
    return when (exponent) {
        0f -> FloatTensor.ones(this.shape)
        1f -> this
        else -> this.operations.pow(this, exponent)
    }
}
