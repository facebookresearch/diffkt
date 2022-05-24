/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.symbolicDiff.logProb

import org.diffkt.*
import org.diffkt.tracing.*

/** A function to be symbolically differentiated. */
abstract class DiffFunction {
    val traceId = TraceId()

    val PI = constant("PI")

    /**
     * Use this to create or define parameters that are not "active variables"
     * for the purpose of computing the derivative.
     */
    fun constant(name: String): DScalar =
        TracingScalar.Variable(0, name, traceId)

    /** The function to be differentiated */
    abstract fun function(value: DScalar): DScalar

    /** Print the code for the primal, first, and second derivatives of the function. */
    fun printCode() {
        // The function
        fun f(input: DScalar): DScalar = function(input)

        // The function and the first derivative of the function
        fun fd(input: DScalar): Pair<DScalar, DScalar> {
            return primalAndForwardDerivative(
                x = input,
                f = ::f,
            )
        }

        // The function and the first and second derivatives of the function
        fun fdd(input: DScalar): List<DScalar> {
            val result = primalAndForwardDerivative(
                x = input,
                f = ::fd,
                extractDerivative = { input: DScalar, d1: Pair<DScalar, DScalar>, extractDerivative: (input: DTensor, output: DTensor) -> DTensor ->
                    extractDerivative(input, d1.second) as DScalar
                },
            )
            return listOf(result.first.first, result.first.second, result.second)
        }

        val input = constant("input")
        val trace = dedag(simplify(fdd(input)), 0, traceId, rewriteVariableReferences = false)

        // Output the generated source code.
        for ((index, expr) in trace.assignments) {
            println("auto t$index = ${(expr as TracingTensor).printedForm(0)};")
        }
        val results = trace.value.map { if (it is TracingScalar) it else TracingTensorOperations.wrap(it) }
        println("auto primal = ${results[0].printedForm(0)}")
        println("auto derivative = ${results[1].printedForm(0)}")
        println("auto derivative2 = ${results[2].printedForm(0)}")
    }
}

fun main() {
    println()
    println("// bernoulli")
    val bernoulli = object : DiffFunction() {
        val logits = constant("logits")
        fun negativeCrossEntropyLossFromLogits(x: DScalar, y: DScalar): DScalar {
            val abs = abs(y)
            val masked = ifThenElse(abs le 88f, y, FloatScalar.ZERO)

            val z = -x * ln(1f + exp(-masked)) - (1f - x) * ln(1f + exp(masked))
            val upperBoundChecked = ifThenElse(y le 88f, z, -(1f - x) * y)
            return ifThenElse(y gt -88f, upperBoundChecked, x * y)
        }
        override fun function(value: DScalar): DScalar {
            return negativeCrossEntropyLossFromLogits(value, logits)
        }
    }
    bernoulli.printCode()

    println()
    println("// beta")
    val beta = object : DiffFunction() {
        val alpha = constant("alpha")
        val beta = constant("beta")
        override fun function(value: DScalar): DScalar {
            val i0 = (alpha - 1f) * ln(value) + (beta - 1f) * ln(1f - value)
            val i1 = lgamma(alpha + beta) - lgamma(alpha) - lgamma(beta)
            return i0 + i1
        }
    }
    beta.printCode()

    println()
    println("// gamma")
    val gamma = object : DiffFunction() {
        val concentration = constant("concentration")
        val rate = constant("rate")

        override fun function(value: DScalar): DScalar {
            return concentration * ln(rate) +
                    (concentration - 1f) * ln(value) -
                    rate * value -
                    lgamma(concentration)
        }
    }
    gamma.printCode()

    println()
    println("// half cauchy")
    val halfCauchy = object : DiffFunction() {
        val scale = constant("scale")
        override fun function(value: DScalar): DScalar {
            val cauchyLogProb = -ln(PI / 2) - ln(scale) - ln(1f + (value / scale).pow(2))
            return ifThenElse(value lt 0f, FloatScalar(Float.NEGATIVE_INFINITY), cauchyLogProb)
        }
    }
    halfCauchy.printCode()

    println()
    println("// half normal")
    val halfNormal = object : DiffFunction() {
        val scale = constant("scale")
        override fun function(value: DScalar): DScalar {
            return ln(sqrt(2.0f / PI)) - ln(scale) - (0.5f * (value/scale).pow(2))
        }
    }
    halfNormal.printCode()

    println()
    println("// normal")
    val normal = object : DiffFunction() {
        val loc = constant("loc")
        val scale = constant("scale")
        fun normalLogProb(value: DScalar, loc: DScalar, scale: DScalar): DScalar {
            val variance = scale.pow(2)
            return -((value - loc).pow(2)) / (2f * variance) - ln(scale) - ln(sqrt(2.0f * PI))
        }
        override fun function(value: DScalar): DScalar {
            return normalLogProb(value, loc, scale)
        }
    }
    normal.printCode()

    println()
    println("// pointMass")
    val pointMass = object : DiffFunction() {
        val alpha = constant("alpha")
        override fun function(value: DScalar): DScalar {
            return ifThenElse(value eq alpha, FloatScalar.ZERO, FloatScalar(Float.NEGATIVE_INFINITY))
        }
    }
    pointMass.printCode()

    println()
    println("// studentT")
    val studentT = object : DiffFunction() {
        val df = constant("df")
        val loc = constant("loc")
        val scale = constant("scale")
        val z = ln(scale) + 0.5f * ln(df) + 0.5f * ln(PI) + lgamma(0.5f * df) - lgamma(0.5f * (df + 1f))
        override fun function(value: DScalar): DScalar {
            val y = (value - loc) / scale
            return -0.5f * (df + 1f) * ln((y.pow(2) / df) + 1f) - z
        }
    }
    studentT.printCode()

    println()
    println("// truncatedCauchy")
    val truncatedCauchy = object : DiffFunction() {
        val loc = constant("loc")
        val scale = constant("scale")
        private val aTan0 = atan((- loc) / scale)
        private val logPDFConstant = -ln(2F / PI - aTan0)

        override fun function(value: DScalar): DScalar {
            return logPDFConstant - ln(scale) - ln(1F + (value - loc / scale).pow(2))
        }
    }
    truncatedCauchy.printCode()

    println()
    println("// uniform")
    val uniform = object : DiffFunction() {
        val low = constant("low")
        val high = constant("high")

        override fun function(value: DScalar): DScalar {
            val lb = low le value
            val ub = high gt value
            return ln(lb * ub) - ln(high - low)
        }
    }
    uniform.printCode()

}
