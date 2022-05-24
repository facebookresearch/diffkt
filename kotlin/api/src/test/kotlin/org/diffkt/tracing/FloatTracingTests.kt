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
import testutils.*
import org.diffkt.random.RandomKey

fun <W : Wrappable<W>> W.floatEval(variables: FloatArray): W {
    val wrapper = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            return when (value) {
                is TracingTensor -> FloatScalar(value.floatEval(variables))
                else -> value
            }
        }

        override fun wrapRandomKey(value: RandomKey): RandomKey {
            TODO("Not yet implemented")
        }
    }

    return wrapper.wrap(this)
}

class FloatTracingTests : AnnotationSpec() {
    // two different functions of two variables
    fun f(x: DScalar, y: DScalar): DScalar = (x * x * y)
    fun g(x: DScalar, y: DScalar): DScalar = (x * y * y)

    data class Inputs(val x: DScalar, val y: DScalar) : Differentiable<Inputs> {
        override fun wrap(wrapper: Wrapper) = Inputs(wrapper.wrap(x), wrapper.wrap(y))
    }
    data class Outputs(val f: DScalar, val g: DScalar) : Differentiable<Outputs> {
        override fun wrap(wrapper: Wrapper) = Outputs(wrapper.wrap(f), wrapper.wrap(g))
    }
    data class Derivatives(val dfdx: DScalar, val dfdy: DScalar, val dgdx: DScalar, val dgdy: DScalar) :
        Differentiable<Derivatives> {
        override fun wrap(wrapper: Wrapper) = Derivatives(wrapper.wrap(dfdx), wrapper.wrap(dfdy), wrapper.wrap(dgdx), wrapper.wrap(dgdy))
    }

    fun function(i: Inputs): Outputs {
        return Outputs(f(i.x, i.y), g(i.x, i.y))
    }
    fun functionForwardDerivatives(i: Inputs): Derivatives {
        val forward = primalAndForwardDerivative(i, ::function, extractDerivative = { input: Inputs, output: Outputs, extractor: (input: DTensor, output: DTensor) -> DTensor ->
            Derivatives(
                dfdx = extractor(input.x, output.f) as DScalar,
                dfdy = extractor(input.y, output.f) as DScalar,
                dgdx = extractor(input.x, output.g) as DScalar,
                dgdy = extractor(input.y, output.g) as DScalar
            )
        }).second
        return forward
    }
    fun functionReverseDerivatives(i: Inputs): Derivatives {
        val reverse = primalAndReverseDerivative(i, ::function, extractDerivative = { input: Inputs, output: Outputs, extractor: (input: DTensor, output: DTensor) -> DTensor ->
            Derivatives(
                dfdx = extractor(input.x, output.f) as DScalar,
                dfdy = extractor(input.y, output.f) as DScalar,
                dgdx = extractor(input.x, output.g) as DScalar,
                dgdy = extractor(input.y, output.g) as DScalar
            )
        }).second
        return reverse
    }

    val x: DScalar = FloatScalar(1.1f)
    val y: DScalar = FloatScalar(-3.2f)
    val inputs = Inputs(x, y)
    val expectedOutputs = Outputs(f(x, y), g(x, y))
    val expectedDerivatives = Derivatives(
        dfdx = forwardDerivative(x) { xx: DScalar -> f(xx, y) },
        dfdy = forwardDerivative(y) { yy: DScalar -> f(x, yy) },
        dgdx = forwardDerivative(x) { xx: DScalar -> g(xx, y) },
        dgdy = forwardDerivative(y) { yy: DScalar -> g(x, yy) },
    )

    @Test
    fun testFunction() {
        function(inputs) shouldBe expectedOutputs
    }

    @Test
    fun testForward() {
        functionForwardDerivatives(inputs) shouldBe expectedDerivatives
    }

    @Test
    fun testReverse() {
        functionReverseDerivatives(inputs) shouldBe expectedDerivatives
    }

    val tid = TraceId()
    val tracingInputs = Inputs(TracingScalar.Variable(0, "x", traceId = tid), TracingScalar.Variable(1, "y", traceId = tid))
    val tracedVariables = run {
        val t = ArrayList<DTensor?>()
        t.add(x)
        t.add(y)
        t.toArray(Array<DTensor?>(2) { null }) as Array<DTensor?>
    }

    val evaluationMap = run {
        val t = ArrayList<Float>(2)
        t.add(x.value)
        t.add(y.value)
        t.toFloatArray()
    }

    @Test
    fun testTracing() {
        val tracingOutputs = function(tracingInputs)
        (tracingOutputs.f as TracingTensor).printedForm() shouldBe "(x * x) * y"
        (tracingOutputs.g as TracingTensor).printedForm() shouldBe "(x * y) * y"
        tracingOutputs.eval(tracedVariables, tid) shouldBe expectedOutputs
    }

    @Test
    fun testEval() {
        val tracingOutputs = function(tracingInputs)
        tracingOutputs.floatEval(evaluationMap) shouldBe expectedOutputs
    }

    @Test
    fun testTracingForward() {
        val tracingForwardDerivatives = simplify(functionForwardDerivatives(tracingInputs))
        (tracingForwardDerivatives.dfdx as TracingTensor).printedForm() shouldBe
                "y * (2.0f * x)"
        (tracingForwardDerivatives.dgdx as TracingTensor).printedForm() shouldBe
                "y * y"
        (tracingForwardDerivatives.dfdy as TracingTensor).printedForm() shouldBe
                "x * x"
        (tracingForwardDerivatives.dgdy as TracingTensor).printedForm() shouldBe
                "(x * y) + (y * x)"
        tracingForwardDerivatives.eval(tracedVariables, tid) shouldBe expectedDerivatives
    }

    fun testEvalForward() {
        val tracingForwardDerivatives = simplify(functionForwardDerivatives(tracingInputs))
        tracingForwardDerivatives.floatEval(evaluationMap) shouldBe expectedDerivatives
    }

    @Test
    fun testTracingReverse() {
        val tracingReverseDerivatives = simplify(functionReverseDerivatives(tracingInputs))
        (tracingReverseDerivatives.dfdx as TracingTensor).printedForm() shouldBe
                "2.0f * (x * y)"
        (tracingReverseDerivatives.dgdx as TracingTensor).printedForm() shouldBe
                "y * y"
        (tracingReverseDerivatives.dfdy as TracingTensor).printedForm() shouldBe
                "x * x"
        (tracingReverseDerivatives.dgdy as TracingTensor).printedForm() shouldBe
                "2.0f * (x * y)"
        tracingReverseDerivatives.eval(tracedVariables, tid) shouldBe expectedDerivatives
    }

    @Test
    fun testEvalReverse() {
        val tracingReverseDerivatives = simplify(functionReverseDerivatives(tracingInputs))
        tracingReverseDerivatives.floatEval(evaluationMap) shouldBe expectedDerivatives
    }
}
