/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.conjugateGradient

interface VectorSpace<Vector, Scalar> {
    operator fun Vector.plus(b: Vector): Vector
    operator fun Vector.minus(b: Vector): Vector
    operator fun Vector.times(b: Scalar): Vector
    operator fun Vector.unaryMinus(): Vector
    fun Vector.dot(b: Vector): Scalar
    operator fun Scalar.compareTo(b: Scalar): Int
    operator fun Scalar.div(b: Scalar): Scalar
    val zeroScalar: Scalar
}