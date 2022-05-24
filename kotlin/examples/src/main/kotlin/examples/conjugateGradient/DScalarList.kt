/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.conjugateGradient

import org.diffkt.*
import examples.utils.conjugateGradient.NewtonCGOptimizer
import org.diffkt.div as diffktDiv

object DScalarListVectorSpace : NewtonCGOptimizer.DifferentiableVectorSpace<List<DScalar>, List<DScalar>, DScalar>() {
    override val tangentVectorSpace = this

    // Because this method overrides two distinct methods in the supertype, we cannot
    // be consistent with the parameter names of both.
    @Suppress("PARAMETER_NAME_CHANGED_ON_OVERRIDE")
    override operator fun List<DScalar>.plus(b: List<DScalar>): List<DScalar> {
        return this.zip(b) { a, bb -> a + bb }
    }

    override operator fun List<DScalar>.minus(b: List<DScalar>): List<DScalar> {
        return this.zip(b) { a, bb -> a - bb }
    }

    override operator fun List<DScalar>.unaryMinus(): List<DScalar> {
        return this.map { xi -> -xi }
    }

    override operator fun List<DScalar>.times(b: DScalar): List<DScalar> {
        return this.map { xi -> b * xi }
    }

    operator fun List<DScalar>.times(b: List<DScalar>): List<DScalar> {
        return this.zip(b) { a, bb -> a * bb }
    }

    fun List<DScalar>.sum(): DScalar {
        return this.reduce(DScalar::plus)
    }

    override fun List<DScalar>.dot(b: List<DScalar>): DScalar {
        return this.zip(b) { a, bb -> a * bb }.reduce(DScalar::plus)
    }

    override operator fun DScalar.compareTo(b: DScalar) = this.basePrimal().value.compareTo(b.basePrimal().value)

    override operator fun DScalar.div(b: DScalar): DScalar = this.diffktDiv(b)

    override val zeroScalar: DScalar = FloatScalar(0f)

    override fun grad(f: (List<DScalar>) -> DScalar, x: List<DScalar>): List<DScalar> {
        return primalAndGradient(x, f).second
    }

    override fun grad(f: (List<DScalar>) -> DScalar): (List<DScalar>) -> List<DScalar> {
        return { x: List<DScalar> ->
            grad(f, x)
        }
    }

    override fun jvp(f: (List<DScalar>) -> List<DScalar>, x: List<DScalar>, v: List<DScalar>): List<DScalar> {
        return primalAndJvp(x, v, f).second
    }

    fun List<DScalar>.extractGradient(
        output: DScalar,
        extractTensorDerivative: (DTensor, DTensor) -> DTensor,
    ): List<DScalar> {
        return this.map { element -> extractTensorDerivative(element, output) as DScalar }
    }
}

fun scalarListOf(vararg elements: Float): List<DScalar> = elements.map { FloatScalar(it) }
fun scalarListOf(vararg elements: DScalar): List<DScalar> = elements.toList()