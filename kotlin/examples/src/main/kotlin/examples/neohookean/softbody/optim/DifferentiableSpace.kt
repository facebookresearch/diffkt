/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.softbody.optim

import org.diffkt.DTensor

interface DifferentiableSpace<Primal, Tangent, Scalar> {
    val tangentVectorSpace: VectorSpace<Tangent, Scalar>
    operator fun Primal.plus(tangent: Tangent): Primal
    fun extractInputTangent(input: Primal, extractTensorTangent: (DTensor) -> DTensor): Tangent
}