/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.symbolicDiff.reverse

import org.diffkt.*
import org.diffkt.tracing.*

/**
 * A program that prints the code needed to implement the pullback in an extension
 * of reverse differentiation designed to compute both the first and second derivatives.
 */
fun main() {
    val traceId = TraceId()
    val _x = TracingScalar.Variable(varIndex = 0, name = "x", traceId)
    val left = TracingScalar.Variable(varIndex = 0, name = "left", traceId)
    val right = TracingScalar.Variable(varIndex = 0, name = "right", traceId)
    val upstream1 = TracingScalar.Variable(varIndex = 0, name = "upstream1", traceId)
    val upstream2 = TracingScalar.Variable(varIndex = 0, name = "upstream2", traceId)

    val x = _x
    fun f(x: DScalar): DScalar {
        return polygamma(10, polygamma(100, x))
    }

    val primal = f(x)
    val pushback1 = jvp(x, upstream1) { xx ->
        f(xx as DScalar)
    } as DScalar
    val pushback2 = vjp(x, upstream2) { xx ->
        jvp(xx, FloatScalar.ONE) { xxx ->
            f(xxx as DScalar)
        }
    } as DScalar

    val result = simplify(listOf(primal, pushback1, pushback2))
    val (assignments, value) = printedForm(dedag(result, 0, traceId, rewriteVariableReferences = false))
    print(assignments)
    println("primal = ${value[0]};")
    println("${x.name}.upstream1 += ${value[1]};")
    println("${x.name}.upstream2 += ${value[2]};")
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
