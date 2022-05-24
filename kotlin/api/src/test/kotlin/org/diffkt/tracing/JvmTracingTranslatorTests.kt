/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe
import org.diffkt.*
import org.diffkt.external.External
import kotlin.math.*
import kotlin.test.assertTrue

class JvmTracingTranslatorTests: AnnotationSpec() {
    @Test
    fun simpleTest() {
        fun f(x: DScalar, y: DScalar) = 2 * x + 3 * y
        val traceId = TraceId()
        val x = TracingScalar.Variable(varIndex = 0, name = "x", traceId = traceId)
        val y = TracingScalar.Variable(varIndex = 1, name = "y", traceId = traceId)
        val fTrace = f(x, y)
        val trace = dedeepen(dedag(fTrace, numInputs = 2, traceId = traceId))
        require(trace.numInputs == 2)
        require(trace.numTemps == 0)
        require(trace.numResults == 1)
        val gen = JvmGenerator<DScalar>(trace)

        val evaluator = gen.getEvaluator()
        val data = FloatArray(trace.numInputs + trace.numTemps + trace.numResults)
        data[0] = 1f
        data[1] = 20f
        evaluator?.invoke(data)
        val result = data[trace.numInputs + trace.numTemps]
        result shouldBe 62f
    }

    val k = 1.1f

    private fun simpleOneInputOneOutputTraceRun(input: Float, numTemps: Int = 0, f: (DScalar) -> DScalar): Float {
        val traceId = TraceId()
        val x = TracingScalar.Variable(varIndex = 0, name = "x", traceId = traceId)
        val fTrace = f(x)
        val trace = dedeepen(dedag(fTrace, numInputs = 1, traceId = traceId))
        require(trace.numInputs == 1)
        require(trace.numTemps == numTemps)
        require(trace.numResults == 1)
        val gen = JvmGenerator<DScalar>(trace)

        val evaluator = gen.getEvaluator()
        val data = FloatArray(trace.numInputs + trace.numTemps + trace.numResults)
        data[0] = input
        evaluator?.invoke(data)
        return data[trace.numInputs + trace.numTemps]
    }

    @Test
    fun `test that sin works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { sin(it) }
        result shouldBe sin(k)
    }

    @Test
    fun `test that cos works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { x -> forwardDerivative(x) { it: DScalar -> sin(it)} }
        result shouldBe cos(k)
    }

    @Test
    fun `test that abs works`() {
        val result = simpleOneInputOneOutputTraceRun(-k, 1) { abs(it) }
        result shouldBe abs(-k)
    }

    @Test
    fun `test that tan works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { tan(it) }
        result shouldBe tan(k)
    }

    @Test
    fun `test that atan works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { atan(it) }
        result shouldBe atan(k)
    }

    @Test
    fun `test that exp works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { exp(it) }
        result shouldBe exp(k)
    }

    @Test
    fun `test that ln works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { ln(it) }
        result shouldBe ln(k)
    }

    @Test
    fun `test that lgamma works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { lgamma(it) }
        result shouldBe External.lgamma(k)
    }

    @Test
    fun `test that digamma works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { digamma(it) }
        result shouldBe External.digamma(k)
    }

    @Test
    fun `test that polygamma works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { polygamma(3, it) }
        result shouldBe External.polygamma(3, k)
    }

    @Test
    fun `test that sqrt works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { sqrt(it) }
        result shouldBe sqrt(k)
    }

    @Test
    fun `test that tanh works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { tanh(it) }
        result shouldBe tanh(k)
    }

    @Test
    fun `test that pow works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { it.pow(3.5f) }
        result shouldBe k.pow(3.5f)
    }

    @Test
    fun `test that sigmoid works`() {
        val result = simpleOneInputOneOutputTraceRun(k) { sigmoid(it) }
        result shouldBe sigmoidElem(k)
    }

    @Test
    fun `test that relu works`() {
        val result1 = simpleOneInputOneOutputTraceRun(Float.NaN) { relu(it) }
        val result2 = simpleOneInputOneOutputTraceRun(Float.NEGATIVE_INFINITY) { relu(it) }
        val result3 = simpleOneInputOneOutputTraceRun(-15.5f) { relu(it) }
        val result4 = simpleOneInputOneOutputTraceRun(0f) { relu(it) }
        val result5 = simpleOneInputOneOutputTraceRun(15.5f) { relu(it) }
        val result6 = simpleOneInputOneOutputTraceRun(Float.POSITIVE_INFINITY) { relu(it) }
        assertTrue { result1.isNaN() }
        result2 shouldBe 0f
        result3 shouldBe 0f
        result4 shouldBe 0f
        result5 shouldBe 15.5f
        result6 shouldBe Float.POSITIVE_INFINITY
    }

    @Test
    fun `test that relugrad works`() {
        val upstream = FloatScalar(10f)
        val result1 = simpleOneInputOneOutputTraceRun(Float.NaN) { reluGrad(it, upstream) }
        val result2 = simpleOneInputOneOutputTraceRun(Float.NEGATIVE_INFINITY) { reluGrad(it, upstream) }
        val result3 = simpleOneInputOneOutputTraceRun(-15.5f) { reluGrad(it, upstream) }
        val result4 = simpleOneInputOneOutputTraceRun(0f) { reluGrad(it, upstream) }
        val result5 = simpleOneInputOneOutputTraceRun(15.5f) { reluGrad(it, upstream) }
        val result6 = simpleOneInputOneOutputTraceRun(Float.POSITIVE_INFINITY) { reluGrad(it, upstream) }
        result1 shouldBe 10f
        result2 shouldBe 0f
        result3 shouldBe 0f
        result4 shouldBe 0f
        result5 shouldBe 10f
        result6 shouldBe 10f
    }
}
