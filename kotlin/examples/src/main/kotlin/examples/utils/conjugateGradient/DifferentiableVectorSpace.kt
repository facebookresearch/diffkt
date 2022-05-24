/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.utils.conjugateGradient

interface DifferentiableVectorSpace<Vector, Tangent, Scalar> : VectorSpace<Vector, Scalar> {
    val tangentVectorSpace: DifferentiableVectorSpace<Tangent, Tangent, Scalar>

    @Suppress("INAPPLICABLE_JVM_NAME")
    @JvmName("VectorPlusTangent")
    operator fun Vector.plus(tangent: Tangent): Vector
}

abstract class DifferentiableVectorSpaceWithSameTangentSpace<Vector, Scalar> :
    DifferentiableVectorSpace<Vector, Vector, Scalar> {
    override val tangentVectorSpace: DifferentiableVectorSpaceWithSameTangentSpace<Vector, Scalar> = this
}