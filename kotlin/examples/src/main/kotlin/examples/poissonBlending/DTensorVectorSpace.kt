/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.poissonBlending

import examples.utils.conjugateGradient.DifferentiableVectorSpace
import examples.utils.conjugateGradient.NewtonCGOptimizer
import org.diffkt.*
import org.diffkt.plus as diffktPlus
import org.diffkt.minus as diffktMinus
import org.diffkt.times as diffktTimes
import org.diffkt.div as diffktDiv
import org.diffkt.unaryMinus as diffktUnaryMinus
import org.diffkt.compareTo as diffktCompareTo

object DTensorVectorSpace: NewtonCGOptimizer.DifferentiableVectorSpace<DTensor, DTensor, DScalar>() {
    override val zeroScalar: DScalar = FloatScalar(0f)

    override fun DTensor.plus(b: DTensor): DTensor = this.diffktPlus(b)

    override fun DTensor.minus(b: DTensor): DTensor = this.diffktMinus(b)

    override fun DTensor.times(b: DScalar): DTensor = this.diffktTimes(b)

    override fun DTensor.unaryMinus(): DTensor = this.diffktUnaryMinus()

    override fun DTensor.dot(b: DTensor): DScalar = (this.diffktTimes(b)).sum()

    override fun DScalar.compareTo(b: DScalar): Int = this.diffktCompareTo(b)

    override fun DScalar.div(b: DScalar): DScalar = this.diffktDiv(b)

    override val tangentVectorSpace: DifferentiableVectorSpace<DTensor, DTensor, DScalar>
        get() = this

    override fun grad(f: (DTensor) -> DScalar, x: DTensor): DTensor {
        return reverseDerivative(x, f)
    }

    override fun jvp(f: (DTensor) -> DTensor, x: DTensor, v: DTensor): DTensor {
        return org.diffkt.jvp(x, v, f)
    }
}
