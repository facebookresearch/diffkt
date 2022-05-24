/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

// ==========
// implementations of tensorOf()
// ==========

/**
 * Create a [FloatTensor] from variable length parameters of type [Float].
 * The [FloatTensor] that is returned is a 1D array.
 */
fun tensorOf(vararg values: Float): FloatTensor {
    // The Kotlin language reference says that values parameter is of type Array<Float>,
    // but in fact it is of type FloatArray.
    return FloatTensor(Shape(values.size), values)
}

/**
 * Create a [FloatTensor] from the list of values given.
 */
fun tensorOf(values: List<Float>): FloatTensor {
    return FloatTensor(Shape(values.count()), values.toFloatArray())
}

/**
 * Create a [DTensor] from variable length parameters of type [DScalar] values.
 */
fun tensorOf(vararg values: DScalar): DTensor = tensorOf(values.toList())

/**
 * Create a [DTensor] from the list of [DScalar].
 */
fun tensorOf(values: List<DScalar>): DTensor = meld(values)
