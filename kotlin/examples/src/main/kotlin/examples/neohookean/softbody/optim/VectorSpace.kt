/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.softbody.optim

interface VectorSpace<Vector, Scalar> {
    operator fun Vector.plus(b: Vector): Vector
    operator fun Vector.times(b: Scalar): Vector
    operator fun Vector.unaryMinus(): Vector
}