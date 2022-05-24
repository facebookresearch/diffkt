/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.symbolicDiff.forward

import org.diffkt.*
import org.diffkt.forward.*
import org.diffkt.tracing.*

/**
 * A program that prints the code needed to implement division in an extension
 * of dual numbers designed to compute both the first and second derivatives.
 */
fun main() {
    val traceId = TraceId()

    fun f(x: DScalar): DScalar {
        return polygamma(10, polygamma(100, x))
    }

    val x = TracingScalar.Variable(varIndex = 0, name = "x", traceId)
    val primal = f(x)
    val xp = TracingScalar.Variable(varIndex = 0, name = "x.tangent1", traceId)
    val derivative1 = jvp(x, xp) { xx ->
        f(xx as DScalar)
    }
    val xpp = TracingScalar.Variable(varIndex = 0, name = "x.tangent2", traceId)
    val derivative2 = jvp(x, xpp) { xx ->
        jvp(xx, xp) { xxx ->
            f(xxx as DScalar)
        }
    }

    val result = simplify(listOf(primal, derivative1, derivative2))
    val (assignments, value) = printedForm(
        dedag(
            result,
            0,
            traceId,
            rewriteVariableReferences = false
        )
    )

    print(assignments)
    println("double newPrimal = ${value[0]};")
    println("double newDerivative1 = ${value[1]};")
    println("double newDerivative2 = ${value[2]};")
}

fun <TValue : Any> printedForm(dedagged: DedaggedTracingTensor<TValue>): Pair<String, TValue> {
    val result = StringBuilder()
    for ((tempIndex, tempValue) in dedagged.assignments) {
        result.appendLine("double t$tempIndex = ${tempValue.rawPrintedForm()};")
    }
    val printingWrapper = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            return PrintedTensor(value.toCodeString(), value.shape)
        }
    }
    return Pair(result.toString(), printingWrapper.wrap(dedagged.value))
}
