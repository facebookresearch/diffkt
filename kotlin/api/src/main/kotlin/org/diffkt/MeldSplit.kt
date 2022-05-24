/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.tracing.TracingTensor

/**
 * Takes a list of values (each of which can be either a [DTensor] or a [DScalar])
 * and combine their values into a single [DTensor] containing all of the values
 * from the inputs.
 */
fun meld(vararg value: DTensor) = meld(value.toList())

/**
 * Return a tensor with the highest derivative ID.
 */
internal fun highestDerivativeID(values: List<DTensor>): DTensor {
    // Determine the highest derivative ID.
    var derivativeID: DerivativeID = NoDerivativeID
    var sample = values.first()
    for (value in values) {
        if (value.derivativeID.sequence > derivativeID.sequence) {
            derivativeID = value.derivativeID
            sample = value
        } else if (derivativeID == NoDerivativeID && value is TracingTensor) {
            sample = value
        }
    }
    return sample
}

internal fun highestDerivativeID(vararg values: DTensor): DTensor {
    return highestDerivativeID(values.toList())
}

/**
 * Takes a list of values (each of which can be either a [DTensor] or a [DScalar])
 * and combine their values into a single [DTensor] containing all of the values
 * from the inputs.
 */
fun meld(values: List<DTensor>): DTensor {
    if (values.isEmpty()) throw IllegalArgumentException("empty input")
    val sample = highestDerivativeID(values)
    return sample.operations.meld(values, sample.derivativeID)
}

/**
 * Takes a tensor, and a list of the desired shapes of the components to
 * split it into, and breaks the values up into a list of [DValue]s.  For
 * a desired shape that is an empty [List], a [DScalar] is produced.
 * Otherwise a [DTensor] is produced with the desired shape.
 *
 * The total number of scalar values required for the outputs
 * collectively must be exactly the same as the number of values in the input
 * tensor, [this].  Otherwise an [IllegalArgumentException] is thrown.
 */
fun DTensor.split(shapes: List<Shape>): List<DTensor> {
    val sizes = shapes.map { it.product() }
    val totalInputSize = this.shape.product()
    val totalOutputSize = sizes.sum()
    if (totalInputSize != totalOutputSize)
        throw IllegalArgumentException("Input size $totalInputSize is not the same as output size $totalOutputSize")
    return this.operations.split(this, shapes)
}

