/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.ints.shouldBeExactly
import io.kotest.matchers.shouldBe
import io.kotest.matchers.shouldNotBe
import org.diffkt.*
import org.diffkt.model.Dense
import org.diffkt.model.Layer
import org.diffkt.model.Model
import org.diffkt.random.DiffktRandom
import org.diffkt.random.RandomKey
import testutils.floats
import testutils.shouldBeExactly
import kotlin.random.Random

class TracingTests : AnnotationSpec() {
    val tid = TraceId()

    fun env(vararg value: DTensor): Array<DTensor?> {
        return Array<DTensor?>(1) { value[it] }
    }

    fun simplifyAndPrint(x: DTensor): String {
        require(x is TracingTensor)
        return simplify(x).printedForm()
    }

    @Test
    fun testSin() {
        fun f(x: DTensor): DTensor {
            return sin(x.pow(2))
        }

        val x = TracingTensor.Variable(0, "x", Shape(2, 2), tid)
        val y = f(x) as TracingTensor
        val s = simplify(y)
        s.printedForm() shouldBe "sin(x.pow(2.0f))"
        val xf = FloatScalar(1.1f)
        s.eval(env(xf), tid) shouldBe sin(xf.pow(2.0f))
    }

    @Test
    fun testForwardDerivativeScalar_01() {
        fun f(x: DTensor): DTensor {
            return sin(x.pow(2))
        }

        val x = TracingScalar.Variable(0, "x", traceId = tid)
        val y = forwardDerivative(x, ::f) as TracingTensor
        val s = simplify(y)
        s.printedForm() shouldBe "cos(x.pow(2.0f)) * (2.0f * x)"
        val xf = FloatScalar(1.1f)
        s.eval(env(xf), tid) shouldBe cos(xf.pow(2.0f)) * (2.0f * xf)
    }

    @Test
    fun testReverseDerivativeScalar_01() {
        fun f(x: DTensor): DTensor {
            return sin(x.pow(2))
        }

        val x = TracingScalar.Variable(0, "x", traceId = tid)
        val y = reverseDerivative(x, ::f)
        simplifyAndPrint(y) shouldBe "cos(x.pow(2.0f)) * (2.0f * x)"
    }

    @Test
    fun testForwardDerivativeScalar_02() {
        fun f(x: DTensor): DTensor {
            return -ln(cos(x))
        }

        val x = TracingScalar.Variable(0, "x", traceId = tid)
        val y = forwardDerivative(x, ::f)
        simplifyAndPrint(y) shouldBe "sin(x) / cos(x)" // tan(x)
    }

    @Test
    fun testDerivativeTensor_01() {
        fun f(x: DTensor): DTensor {
            return sin(x.pow(2))
        }

        val x = TracingTensor.Variable(0, "x", Shape(2, 3), traceId = tid)
        simplifyAndPrint(f(x)) shouldBe "sin(x.pow(2.0f))"
        val yf = forwardDerivative(x, ::f)
        simplifyAndPrint(yf) shouldBe
                "((2.0f * x.reshape(Shape(2, 3, 1, 1))) * StridedFloatTensor.identityGradient(Shape(2, 3))) * cos(x.pow(2.0f)).reshape(Shape(2, 3, 1, 1))"
        val yr = reverseDerivative(x, ::f)
        simplifyAndPrint(yr) shouldBe
                "(2.0f * x.reshape(Shape(2, 3, 1, 1))) * (StridedFloatTensor.identityGradient(Shape(2, 3)) * cos(x.pow(2.0f)).reshape(Shape(2, 3, 1, 1)))"
    }

    @Test
    fun testDerivativeTensor_02() {
        fun f(x: DTensor): DTensor {
            return sin(x.pow(2)).sum()
        }

        val x = TracingTensor.Variable(0, "x", Shape(2, 3), traceId = tid)
        simplifyAndPrint(f(x)) shouldBe "sin(x.pow(2.0f)).sum()"
        val yf = forwardDerivative(x, ::f)
        simplifyAndPrint(yf) shouldBe
                "(((2.0f * x.reshape(Shape(2, 3, 1, 1))) * StridedFloatTensor.identityGradient(Shape(2, 3))) * cos(x.pow(2.0f)).reshape(Shape(2, 3, 1, 1))).sum(intArrayOf(0, 1), keepDims = false)"
        val yr = reverseDerivative(x, ::f)
        simplifyAndPrint(yr) shouldBe "(2.0f * x) * cos(x.pow(2.0f))"
    }

    @Test
    fun testDerivativeTensor_03() {
        fun f(x: DTensor): DTensor {
            x as DScalar
            return tensorOf(sin(x), cos(x))
        }

        val x: DTensor = TracingScalar.Variable(0, "x", traceId = tid)
        simplifyAndPrint(f(x)) shouldBe "meld(sin(x), cos(x))"
        val yf = forwardDerivative(x, ::f)
        simplifyAndPrint(yf) shouldBe "meld(cos(x), -sin(x))"
        val yr = reverseDerivative(x, ::f)
        simplifyAndPrint(yr) shouldBe
                "(-(sin(x) * tensorOf(0.0f, 1.0f))) + (cos(x) * tensorOf(1.0f, 0.0f))"
    }

    @Test
    fun testDerivativeTensor_04() {
        fun f(x: DTensor): DTensor {
            return sin(x[0]) + cos(x[1])
        }

        val x: DTensor = TracingTensor.Variable(0, "x", Shape(2), traceId = tid)
        simplifyAndPrint(f(x)) shouldBe "sin(x[0]) + cos(x[1])"
        val yf = forwardDerivative(x, ::f) as TracingTensor
        val sf = simplify(yf)
        sf.printedForm() shouldBe "(cos(x[0]) * tensorOf(1.0f, 0.0f)) - (sin(x[1]) * tensorOf(0.0f, 1.0f))"
        val xf = tensorOf(1.1f, 2.2f)
        sf.eval(env(xf), tid) shouldBe forwardDerivative(xf, ::f)
        sf.eval(env(xf), tid) shouldBe run {
            (cos(xf[0]) * tensorOf(1.0f, 0.0f)) + ((-sin(xf[1])) * tensorOf(0.0f, 1.0f))
        }

        val yr = reverseDerivative(x, ::f) as TracingTensor
        val sr = simplify(yr)
        sr.printedForm() shouldBe "concat(listOf(tensorOf(0.0f), (-sin(x[1])).unsqueeze(axis = 0)), axis = 0) + concat(listOf(cos(x[0]).unsqueeze(axis = 0), tensorOf(0.0f)), axis = 0)"
        sr.eval(env(xf), tid) shouldBe run {
            concat(listOf(tensorOf(0.0f), (-sin(xf[1])).unsqueeze(axis = 0)), axis = 0) + concat(listOf(cos(xf[0]).unsqueeze(axis = 0), tensorOf(0.0f)), axis = 0)
        }
        sf.eval(env(xf), tid) shouldBe reverseDerivative(xf, ::f)
    }

    @Test
    fun testForwardDerivativeMultivariant_01() {
        fun f(x: DTensor, y: DTensor): DTensor {
            return x * x * y
        }

        val x: DTensor = TracingScalar.Variable(0, "x", traceId = tid)
        val y: DTensor = TracingScalar.Variable(1, "y", traceId = tid)
        val pafd = primalAndForwardDerivative<Pair<DTensor, DTensor>, DTensor, Pair<DTensor, DTensor>>(
            x = Pair(x, y),
            f = { xx: Pair<DTensor, DTensor> -> f(xx.first, xx.second) },
            extractDerivative = { extractInput: Pair<DTensor, DTensor>,
                                  extractOutput: DTensor,
                                  extractor: (input: DTensor, output: DTensor) -> DTensor ->
                Pair(extractor(extractInput.first, extractOutput), extractor(extractInput.second, extractOutput))
            })
        val (temps, pafdo) = tracingPrintedForm(simplify(pafd), 2)
        temps shouldBe
                "val t2 = x * x\n"
        val primal = pafdo.first
        (primal as PrintedTensor).printed shouldBe
                "y * t2"
        val (d1, d2) = pafdo.second
        (d1 as PrintedTensor).printed shouldBe
                "y * (2.0f * x)"
        (d2 as PrintedTensor).printed shouldBe
                "t2"
    }

    @Test
    fun testReverseDerivativeMultivariant_01() {
        fun f(x: DTensor, y: DTensor): DTensor {
            return x * x * y
        }

        val x: DTensor = TracingScalar.Variable(0, "x", traceId = tid)
        val y: DTensor = TracingScalar.Variable(1, "y", traceId = tid)
        val pafd = primalAndReverseDerivative<Pair<DTensor, DTensor>, DTensor, Pair<DTensor, DTensor>>(
            x = Pair(x, y),
            f = { xx: Pair<DTensor, DTensor> -> f(xx.first, xx.second) },
            extractDerivative = { extractInput: Pair<DTensor, DTensor>,
                                  extractOutput: DTensor,
                                  extractor: (input: DTensor, output: DTensor) -> DTensor ->
                Pair(extractor(extractInput.first, extractOutput), extractor(extractInput.second, extractOutput))
            })
        val (assignments, pafdo) = tracingPrintedForm(simplify(pafd), 2)
        assignments shouldBe
                "val t2 = x * x\n" +
                "val t3 = x * y\n"
        val primal = pafdo.first
        (primal as PrintedTensor).printed shouldBe
                "y * t2"
        val (d1, d2) = pafdo.second
        (d1 as PrintedTensor).printed shouldBe
                "t3 + t3"
        (d2 as PrintedTensor).printed shouldBe
                "t2"
    }

    fun testAtan() {
        fun f(x: DTensor): DTensor {
            return atan(x.pow(2))
        }

        val x = TracingTensor.Variable(0, "x", Shape(2, 2), tid)
        val y = f(x) as TracingTensor
        val s = simplify(y)
        s.printedForm() shouldBe "atan(x.pow(2.0f))"
        val xf = FloatScalar(1.1f)
        s.eval(env(xf), tid) shouldBe atan(xf.pow(2.0f))
    }

    @Test
    fun testAtanForwardDerivativeScalar() {
        fun f(x: DTensor): DTensor {
            return atan(x.pow(2))
        }

        val x = TracingScalar.Variable(0, "x", traceId = tid)
        val y = forwardDerivative(x, ::f) as TracingTensor
        val s = simplify(y)
        s.printedForm() shouldBe "(1.0f / (x.pow(4.0f) + 1.0f)) * (2.0f * x)"
        val xf = FloatScalar(1.1f)
        s.eval(env(xf), tid) shouldBe (2.0f * xf) / (1f + xf.pow(4))
    }

    @Test
    fun testAtanReverseDerivativeScalar() {
        fun f(x: DTensor): DTensor {
            return atan(x.pow(2))
        }

        val x = TracingScalar.Variable(0, "x", traceId = tid)
        val y = reverseDerivative(x, ::f)
        simplifyAndPrint(y) shouldBe "(1.0f / (x.pow(4.0f) + 1.0f)) * (2.0f * x)"
    }

    @Test
    fun testAtanDerivativeTensor() {
        fun f(x: DTensor): DTensor {
            return atan(x.pow(2))
        }

        val x = TracingTensor.Variable(0, "x", Shape(2, 3), traceId = tid)
        simplifyAndPrint(f(x)) shouldBe "atan(x.pow(2.0f))"
        val yf = forwardDerivative(x, ::f)
        simplifyAndPrint(yf) shouldBe
                "((2.0f * x.reshape(Shape(2, 3, 1, 1))) * StridedFloatTensor.identityGradient(Shape(2, 3))) / (x.pow(4.0f) + tensorOf(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f).reshape(Shape(2, 3))).reshape(Shape(2, 3, 1, 1))"
        val yr = reverseDerivative(x, ::f)
        simplifyAndPrint(yr) shouldBe
                "(2.0f * x.reshape(Shape(2, 3, 1, 1))) * (StridedFloatTensor.identityGradient(Shape(2, 3)) / (x.pow(4.0f) + tensorOf(1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f).reshape(Shape(2, 3))).reshape(Shape(2, 3, 1, 1)))"
    }

    @Test
    fun `test jitting of data-dependent control-flow`() {
        fun f(it: Boolean) = if (it) 1f else 2f

        // The non-jitted function depends on the boolean parameter.
        f(true) shouldBe 1
        f(false) shouldBe 2

        // The jitted function does too.
        val jittedF1 = jit(::f)
        jittedF1(true) shouldBe 1
        jittedF1(false) shouldBe 2

        val jittedF2 = jit(::f)
        jittedF2(false) shouldBe 2
        jittedF2(true) shouldBe 1
    }
    @Test fun `test jitting of control-flow dependent on captured data`() {
        var captured = true
        @Suppress("UNUSED_PARAMETER")
        fun f(it: Boolean) = if (captured) 1f else 2f

        // The non-jitted function depends on the captured boolean.
        captured = true
        f(true) shouldBe 1
        f(false) shouldBe 1
        captured = false
        f(true) shouldBe 2
        f(false) shouldBe 2

        // Sadly, the jitted function doesn't
        val jittedF = jit(::f)
        captured = true
        jittedF(true) shouldBe 1
        captured = false
        jittedF(true) shouldBe 1

        captured = false
        jittedF(false) shouldBe 2
        captured = true
        jittedF(false) shouldBe 2
    }

    @Test
    fun `test jitting of the reverse derivative of a function of one variable`() {
        fun f(x: DScalar): DScalar {
            return sin(x.pow(2))
        }

        fun fp(x: DScalar): DScalar {
            return reverseDerivative(x, ::f)
        }

        val jittedFp = jit(::fp)
        val x = TracingScalar.Variable(0, "x", traceId = tid)
        val fpx = jittedFp(x) as TracingTensor
        fpx.printedForm() shouldBe "cos(x.pow(2.0f)) * (2.0f * x)"
    }

    @Test
    fun `test jitting of the forward derivative of a function of one variable`() {
        fun f(x: DScalar): DScalar {
            return sin(x.pow(2))
        }

        fun fp(x: DScalar): DScalar {
            return forwardDerivative(x, ::f)
        }

        val jittedFp = jit(::fp)
        val x = TracingScalar.Variable(0, "x", traceId = tid)
        val fpx = jittedFp(x) as TracingTensor
        fpx.printedForm() shouldBe "cos(x.pow(2.0f)) * (2.0f * x)"
    }

    @Test
    fun `test jitting of the reverse derivative of a function of two variables`() {
        fun f(x: DScalar, y: DScalar): DScalar {
            return sin(x.pow(2)) + cos(y.pow(3))
        }
        fun fp(l: List<DScalar>): List<DScalar> {
            return primalAndReverseDerivative(l) { ll: List<DScalar> ->
                f(ll[0], ll[1])
            }.second
        }
        val jittedFp0 = jit(::fp)
        val jittedFp = { x: DScalar, y: DScalar -> jittedFp0(listOf(x, y)) }

        val x = TracingScalar.Variable(0, "x", traceId = tid)
        val y = TracingScalar.Variable(1, "y", traceId = tid)
        val (fpx, fpy) = jittedFp(x, y)
        fpx as TracingTensor; fpy as TracingTensor
        fpx.printedForm() shouldBe "cos(x.pow(2.0f)) * (2.0f * x)"
        fpy.printedForm() shouldBe "-((3.0f * y.pow(2.0f)) * sin(y.pow(3.0f)))"
    }

    @Test
    fun `test jitting of the forward derivative of a function of two variables`() {
        fun f(x: DScalar, y: DScalar): DScalar {
            return sin(x.pow(2)) + cos(y.pow(3))
        }
        fun fp(l: List<DScalar>): List<DScalar> {
            return primalAndForwardDerivative(l) { ll: List<DScalar> ->
                f(ll[0], ll[1])
            }.second
        }
        val jittedFp0 = jit(::fp)
        val jittedFp = { x: DScalar, y: DScalar -> jittedFp0(listOf(x, y)) }

        val x = TracingScalar.Variable(0, "x", traceId = tid)
        val y = TracingScalar.Variable(1, "y", traceId = tid)
        val (fpx, fpy) = jittedFp(x, y)
        fpx as TracingTensor; fpy as TracingTensor
        fpx.printedForm() shouldBe "cos(x.pow(2.0f)) * (2.0f * x)"
        fpy.printedForm() shouldBe "-(sin(y.pow(3.0f)) * (3.0f * y.pow(2.0f)))"
    }

    @Test
    fun `test jitting of the reverse derivative of a function returning two results`() {
        fun f(x: DScalar): List<DScalar> {
            return listOf(sin(x.pow(2)), cos(x.pow(3)))
        }
        fun fp(x: DScalar): List<DScalar> {
            return primalAndReverseDerivative<DScalar, List<DScalar>, List<DScalar>>(
                x = x,
                f = ::f,
                extractDerivative = { input: DScalar, output: List<DScalar>, extractor: (DTensor, DTensor)-> DTensor ->
                    output.map { extractor(input, it) as DScalar }
                }).second
        }
        val jittedFp0 = jit(::fp)
        val jittedFp = { x: DScalar -> jittedFp0(x) }

        val x = TracingScalar.Variable(0, "x", traceId = tid)
        val (fp1, fp2) = jittedFp(x)
        fp1 as TracingTensor; fp2 as TracingTensor
        fp1.printedForm() shouldBe
                "(2.0f * x) * cos(x.pow(2.0f))"
        fp2.printedForm() shouldBe
                "-((3.0f * x.pow(2.0f)) * sin(x.pow(3.0f)))"
    }

    @Test
    fun `test jitting of the forward derivative of a function returning two results`() {
        fun f(x: DScalar): List<DScalar> {
            return listOf(sin(x.pow(2)), cos(x.pow(3)))
        }
        fun fp(x: DScalar): List<DScalar> {
            return primalAndForwardDerivative<DScalar, List<DScalar>, List<DScalar>>(
                x = x,
                f = ::f,
                extractDerivative = { input: DScalar, output: List<DScalar>, extractor: (DTensor, DTensor)-> DTensor ->
                    output.map { extractor(input, it) as DScalar }
                }).second
        }
        val jittedFp0 = jit(::fp)
        val jittedFp = { x: DScalar -> jittedFp0(x) }

        val x = TracingScalar.Variable(0, "x", traceId = tid)
        val (fp1, fp2) = jittedFp(x)
        fp1 as TracingTensor; fp2 as TracingTensor
        fp1.printedForm() shouldBe
                "cos(x.pow(2.0f)) * (2.0f * x)"
        fp2.printedForm() shouldBe
                "-((3.0f * x.pow(2.0f)) * sin(x.pow(3.0f)))"
    }

    @Test fun scatter2D() {
        val x = FloatTensor(Shape(2, 5), floats(10))

        jit { xx: DTensor -> xx.scatter(listOf(1, 2), axis = 0, Shape(3, 5)) }(x) shouldBe FloatTensor(
            Shape(3, 5),
            0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)

        jit { xx: DTensor -> xx.scatter(listOf(0, 1, 2), axis = 1, Shape(3, 5)) }(x) shouldBe FloatTensor(
            Shape(3, 5),
            1f, 2f, 3f, 0f, 0f, 6f, 7f, 8f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
    }

    @Test fun gather2D() {
        val x = FloatTensor(Shape(2, 3), floats(6))

        jit { xx: DTensor -> xx.gather(listOf(1), axis = 0) }(x) shouldBe FloatTensor(Shape(1, 3), 4f, 5f, 6f)
        jit { xx: DTensor -> xx.gather(listOf(0, 2), axis = 1) }(x) shouldBe FloatTensor(Shape(2, 2), 1f, 3f, 4f, 6f)
        jit { xx: DTensor -> xx.gather(listOf(0, 2, 1, 0), axis = 1) }(x) shouldBe FloatTensor(Shape(2, 4), 1f, 3f, 2f, 1f, 4f, 6f, 5f, 4f)
    }

    @Test fun `test that we use semantic equality on a model in the jit cache`() {
        val random = Random(123)

        class MyModel(
            val inputs: Int = 4, val middle: Int = 2, val outputs: Int = 1,
            override val layers: List<Layer<*>> = listOf(Dense(inputs, middle, random), Dense(middle, outputs, random))
        ) : Model<MyModel>() {
            override fun hashCode(): Int = combineHash("MyModel")
            override fun equals(other: Any?): Boolean = other is MyModel && other.layers == layers
            override fun withLayers(newLayers: List<Layer<*>>) = MyModel(inputs, middle, outputs, newLayers)
        }

        val model = MyModel()
        @Suppress("UNUSED_PARAMETER")
        fun f(m: MyModel): DScalar {
            return FloatScalar.ONE
        }
        val jittedF = jit(::f)
        fun counter() = jittedF.cacheSize
        counter() shouldBeExactly 0
        jittedF(model)
        counter() shouldBeExactly 1
        jittedF(model)
        counter() shouldBeExactly 1
        val model2 = MyModel() // same shape but different numbers
        jittedF(model2)
        counter() shouldBeExactly 1
        jittedF(model2)
        counter() shouldBeExactly 1
        val model3 = MyModel(5, 2, 1) // different shape
        jittedF(model3)
        counter() shouldBeExactly 2
        jittedF(model3)
        counter() shouldBeExactly 2
    }

    @Test fun `test that dedeepen does its job`() {
        val tid = TraceId()
        val one = TracingScalar.Variable(0, "1", traceId = tid)
        fun sum1(n: Int): DScalar = if (n <= 1) one else sum1(n-1) + FloatScalar(n.toFloat())
        fun sum2(n: Int): DScalar = if (n <= 1) one else FloatScalar(n.toFloat()) + sum2(n-1)
        val variables = listOf(FloatScalar.ONE)
        val n = 10
        val expectedResult = FloatScalar((n * (n + 1)) / 2f)

        val e1 = sum1(n)
        e1.toString() shouldBe "((((((((1 + 2.0f) + 3.0f) + 4.0f) + 5.0f) + 6.0f) + 7.0f) + 8.0f) + 9.0f) + 10.0f"
        val d1 = DedaggedTracingTensor<DScalar>(1, 0, 1, emptyList(), e1, tid, true)
        eval(d1, variables) shouldBe expectedResult
        scalarEval(d1, variables) shouldBe expectedResult
        jitEval(d1, variables) shouldBe expectedResult
        tracingPrintedForm(d1).toString() shouldBe "(, ((((((((1 + 2.0f) + 3.0f) + 4.0f) + 5.0f) + 6.0f) + 7.0f) + 8.0f) + 9.0f) + 10.0f)"
        val d1d = dedeepen(d1, 5)
        eval(d1d, variables) shouldBe expectedResult
        scalarEval(d1d, variables) shouldBe expectedResult
        jitEval(d1d, variables) shouldBe expectedResult
        tracingPrintedForm(d1d).toString() shouldBe "(val t1 = (((1 + 2.0f) + 3.0f) + 4.0f) + 5.0f\n" +
                "val t2 = (((t1 + 6.0f) + 7.0f) + 8.0f) + 9.0f\n" +
                ", t2 + 10.0f)"

        val e2 = sum2(n)
        e2.toString() shouldBe "10.0f + (9.0f + (8.0f + (7.0f + (6.0f + (5.0f + (4.0f + (3.0f + (2.0f + 1))))))))"
        val d2 = DedaggedTracingTensor<DScalar>(1, 0, 1, emptyList(), e2, tid, true)
        eval(d2, variables) shouldBe expectedResult
        scalarEval(d2, variables) shouldBe expectedResult
        jitEval(d2, variables) shouldBe expectedResult
        tracingPrintedForm(d2).toString() shouldBe "(, 10.0f + (9.0f + (8.0f + (7.0f + (6.0f + (5.0f + (4.0f + (3.0f + (2.0f + 1)))))))))"
        val d2d = dedeepen(d2, 5)
        eval(d2d, variables) shouldBe expectedResult
        scalarEval(d2d, variables) shouldBe expectedResult
        jitEval(d2d, variables) shouldBe expectedResult
        tracingPrintedForm(d2d).toString() shouldBe "(val t1 = 5.0f + (4.0f + (3.0f + (2.0f + 1)))\n" +
                "val t2 = 9.0f + (8.0f + (7.0f + (6.0f + t1)))\n" +
                ", 10.0f + t2)"
    }

    @Test fun `test for stack overflow in simplify and dedag and the evaluators`() {
        val moreStackThanReasonable = 60000
        val reasonableStack = 500
        val tid = TraceId()

        fun makeLeaf(): TracingScalar =
            TracingScalar.Plus(TracingScalar.Variable(0, "x", traceId = tid), TracingScalar.Constant(FloatScalar.ONE))
        fun makeDeepSum1(n: Int): TracingScalar =
            (1 until n).fold(makeLeaf()) { a, _ ->
                TracingScalar.Plus(a, makeLeaf())
            }
        val e1 = makeDeepSum1(moreStackThanReasonable)
        val d1 = dedag(simplify(e1), 1, tid)
        d1.assignments.size shouldBe 1
        val d1d = dedeepen(d1, reasonableStack)
        d1d.assignments.size shouldBe 1 + moreStackThanReasonable/reasonableStack
        val variables = listOf<DTensor>(FloatScalar.ZERO)
        eval(d1d, variables) shouldBe FloatScalar(moreStackThanReasonable.toFloat())
        scalarEval(d1d, variables) shouldBe FloatScalar(moreStackThanReasonable.toFloat())

        fun makeDeepSum2(n: Int): TracingScalar =
            (1 until n).fold(makeLeaf()) { a, _ ->
                TracingScalar.Plus(makeLeaf(), a)
            }
        val e2 = makeDeepSum2(moreStackThanReasonable)
        val d2 = dedag(simplify(e2), 1, tid)
        d2.assignments.size shouldBe 1
        val d2d = dedeepen(d2, reasonableStack)
        d2d.assignments.size shouldBe 1 + moreStackThanReasonable/reasonableStack
        eval(d2d, variables) shouldBe FloatScalar(moreStackThanReasonable.toFloat())
        scalarEval(d2d, variables) shouldBe FloatScalar(moreStackThanReasonable.toFloat())
    }

    @Test fun `test for stack overflow in the jit evaluator`() {
        // This is the approximate limit for how large a method we can produce.
        val moreStackThanReasonable = 15000
        val reasonableStack = 50
        val tid = TraceId()

        fun makeLeaf(): TracingScalar =
            TracingScalar.Plus(TracingScalar.Variable(0, "x", traceId = tid), TracingScalar.Constant(FloatScalar.ONE))
        fun makeDeepSum1(n: Int): TracingScalar =
            (1 until n).fold(makeLeaf()) { a, _ ->
                TracingScalar.Plus(a, makeLeaf())
            }
        val e1 = makeDeepSum1(moreStackThanReasonable)
        val d1 = dedag(simplify(e1), 1, tid)
        val d1d = dedeepen(d1, reasonableStack)
        val variables = listOf<DTensor>(FloatScalar.ZERO)
        scalarEval(d1d, variables) shouldBe FloatScalar(moreStackThanReasonable.toFloat())
        jitEval(d1d, variables) shouldBe FloatScalar(moreStackThanReasonable.toFloat())

        fun makeDeepSum2(n: Int): TracingScalar =
            (1 until n).fold(makeLeaf()) { a, _ ->
                TracingScalar.Plus(makeLeaf(), a)
            }
        val e2 = makeDeepSum2(moreStackThanReasonable)
        val d2 = dedag(simplify(e2), 1, tid)
        val d2d = dedeepen(d2, reasonableStack)
        scalarEval(d2d, variables) shouldBe FloatScalar(moreStackThanReasonable.toFloat())
        jitEval(d2d, variables) shouldBe FloatScalar(moreStackThanReasonable.toFloat())
    }

    @Test fun `test that the jit properly handles nested invocations`() {
        val f = jit {
            outerInput: DScalar ->
            val g = jit {
                innerInput: DScalar ->
                outerInput + innerInput
            }
            g(FloatScalar(100f))
        }
        val r = f(FloatScalar(1f))
        r shouldBe FloatScalar(101f)
    }

    @Test fun `test that the scalar jit properly handles nested invocations`() {
        val f = jit(evaluatorToUse = JitEvaluatorToUse.Scalar, f = {
                outerInput: DScalar ->
            val g = jit {
                    innerInput: DScalar ->
                outerInput + innerInput + outerInput
            }
            g(FloatScalar(100f))
        })
        val r = f(FloatScalar(1f))
        r shouldBe FloatScalar(102f)
    }

    @Test fun `test that wrapInput functions works`() {
        class Input(val value: DScalar, val dependencies: List<DScalar>)

        fun f(input: Input): DScalar {
            require(input.dependencies.size == 1)
            val depValue = input.dependencies.first()
            if (depValue is TracingScalar) {
                return input.value * depValue
            }
            return FloatScalar(0F)
        }

        val x = Input(FloatScalar(2F), listOf(FloatScalar(3F)))

        val noWrappedJittedF = jit(::f)
        val noWrappedF = noWrappedJittedF(x)
        noWrappedF shouldBe FloatScalar(0F)

        val wrappedJittedF = jit(::f, wrapInput = {input, wrapper -> Input(input.value, input.dependencies.map{ wrapper.wrap(it) })})
        val wrappedF = wrappedJittedF(x)
        wrappedF shouldBe FloatScalar(6F)
    }

    val sampleValues = listOf(
        Float.NaN, Float.NEGATIVE_INFINITY, -Float.MAX_VALUE, -10f, -Float.MIN_VALUE,
        0f, Float.MIN_VALUE, 10f, Float.MAX_VALUE, Float.POSITIVE_INFINITY
    ).map { FloatScalar(it) }

    @Test fun `test comparison in the jvm generator`() {
        for (comparison in ComparisonKind.values()) {
            fun f(x: DScalar, y: DScalar) = compare(x, y, comparison)
            fun f2(p: Pair<DScalar, DScalar>) = f(p.first, p.second)
            val jittedF = jit(::f2, evaluatorToUse = JitEvaluatorToUse.Jvm)
            fun jf(x: DScalar, y: DScalar) = jittedF(Pair(x, y))
            for (v1 in sampleValues) {
                for (v2 in sampleValues) {
                    f(v1, v2) shouldBe jf(v1, v2)
                }
            }
        }
    }

    @Test fun `test ifThenElse over non-comparison in the jvm generator`() {
        fun f(x: DScalar) = ifThenElse(x, FloatScalar.ONE, FloatScalar.ZERO)
        val jf = jit(::f, evaluatorToUse = JitEvaluatorToUse.Jvm)
        for (comparison in sampleValues) {
            f(comparison) shouldBe jf(comparison)
        }
    }

    @Test fun `test ifThenElse over comparison in the jvm generator`() {
        for (comparison in ComparisonKind.values()) {
            fun f(x: DScalar, y: DScalar) =
                ifThenElse(compare(x, y, comparison), FloatScalar.ONE, FloatScalar.ZERO)
            fun f2(p: Pair<DScalar, DScalar>) = f(p.first, p.second)
            val jittedF = jit(::f2, evaluatorToUse = JitEvaluatorToUse.Jvm)
            fun jf(x: DScalar, y: DScalar) = jittedF(Pair(x, y))
            for (v1 in sampleValues) {
                for (v2 in sampleValues) {
                    f(v1, v2) shouldBe jf(v1, v2)
                }
            }
        }
    }

    @Test fun `test that multiplication by minus one is negation`() {
        val x = TracingScalar.Variable(0, "x", traceId = tid)
        simplify((-1f * x) as TracingTensor).printedForm() shouldBe "-x"
        simplify((-1f * (-1f * x)) as TracingTensor).printedForm() shouldBe "x"
    }

    @Test fun `test that concat and meld work with mixed tracing and float tensors`() {
        val t = tensorOf(*floats(5))
        fun f1(x: DTensor) = meld(t, t, x)
        fun f2(x: DTensor) = concat(listOf(t, t, x))
        fun f3(x: DTensor) = meld(x, x, t)
        fun f4(x: DTensor) = concat(listOf(x, x, t))
        jit(::f1)(t) shouldBeExactly f1(t)
        jit(::f2)(t) shouldBeExactly f2(t)
        jit(::f3)(t) shouldBeExactly f3(t)
        jit(::f4)(t) shouldBeExactly f4(t)
    }

    @Test fun `test tracing random key`() {
        fun f(r: RandomKey): DTensor {
            return r.floats(Shape(2, 2)) + 1f
        }

        val x1 = RandomKey(0)
        val s = jit(::f)
        val res = s(x1)
        res shouldBeExactly f(RandomKey(0))
    }

    @Test fun `test tracing random key on different seeds`() {
        fun f(r: RandomKey): DTensor {
            return r.floats(Shape( 2, 2)) + 1f
        }

        val s = jit(::f)
        val res1 = s(RandomKey(0))
        val res2 = s(RandomKey(1))
        res1 shouldBeExactly f(RandomKey(0))
        res2 shouldBeExactly f(RandomKey(1))
        res1 shouldNotBe res2
    }

    @Test fun `test tracing random key with split in function`() {
        fun f(r: RandomKey): DTensor {
            val splits = r.split(2)
            val res1 = splits.first().floats(2)
            val res2 = splits.last().floats(2)
            // Note: The evidence we have for (res1 != res2) when the jit is evaluated is just that the key isn't reused.
            // Can also manually check this by breakpointing or printing during tests
            return res1 + res2
        }

        val s = jit(::f)
        val res1 = s(RandomKey(0))
        val res2 = s(RandomKey(1))
        res1 shouldBeExactly f(RandomKey(0))
        res2 shouldBeExactly f(RandomKey(1))
        res1 shouldNotBe res2
    }

    @Test fun `test function returning a random key`() {
        fun f(r: RandomKey): RandomKey {
            val splits = r.split(2)
            return splits[1]
        }

        val x1 = RandomKey(0)
        val x2 = RandomKey(1)
        val s = jit(::f)
        val resultKey1 = s(x1)
        val resultKey2 = s(x2)
        resultKey1 shouldNotBe x1
        resultKey2 shouldNotBe x2
        resultKey1 shouldNotBe resultKey2
        assert(resultKey1 !is TracingRandomKey && resultKey2 !is TracingRandomKey)
    }

    @Test fun `test function using a DiffktRandom`() {
        fun f(d: DiffktRandom): DTensor {
            return d.nextUniform(Shape(3))
        }
        val s = jit(::f)
        val x1 = DiffktRandom()
        val x2 = DiffktRandom.fromTimeOfDay()
        s.cacheSize shouldBeExactly 0
        val result1 = s(x1)
        s.cacheSize shouldBeExactly 1
        val result2 = s(x2)
        s.cacheSize shouldBeExactly 1
        assert(result1 != result2 && result1 is DTensor && result2 is DTensor)
    }

    @Test fun `test function returning a DiffktRandom`() {
        fun f(d: DiffktRandom): DiffktRandom {
            return DiffktRandom(d.getRandomKey())
        }
        val s = jit(::f)
        val x1 = DiffktRandom(RandomKey().permitReuse())
        val x2 = DiffktRandom(x1.randomKey)
        val result1 = s(x1)
        val result2 = f(x2)
        assert(result1 == result2)
    }

    @Test fun `test class that contains DiffktRandom`() {
        class Foo(val d: DiffktRandom): Wrappable<Foo> {
            override fun wrap(wrapper: Wrapper): Foo {
                return Foo(wrapper.wrap(d))
            }
            fun doSomething(): DTensor {
                return d.nextUniform(Shape(3))
            }
        }
        fun f(x: Foo): DTensor {
            return x.doSomething()
        }
        val s = jit(::f)
        val x1 = Foo(DiffktRandom(RandomKey().permitReuse()))
        val x2 = Foo(DiffktRandom(x1.d.randomKey))
        val result1 = s(x1)
        val result2 = f(x2)
        result1 shouldBeExactly result2
    }

    @Test fun `test jitting derivative of a function that takes in a DiffktRandom`() {
        fun f(x: Pair<DScalar, DiffktRandom>): DScalar {
            return x.first * x.second.nextUniform()
        }
        fun fp(xp: Pair<DScalar, DiffktRandom>): DScalar {
            return forwardDerivative(xp, ::f) { input: Pair<DScalar, DiffktRandom>, output: DScalar, extractOneDerivative: (input: DTensor, output: DTensor) -> DTensor ->  extractOneDerivative(input.first, output)} as DScalar
        }
        val x = ::fp
        val s = jit(::fp)
        val d1 = DiffktRandom(RandomKey().permitReuse())
        val d2 = DiffktRandom(d1.randomKey)
        val result = s(Pair(FloatScalar(1f), d1))
        result shouldBeExactly d2.nextUniform()
    }

    @Test fun `test that the jit is fully evaluated on nested invocations with random keys`() {
        val f = jit {
                outerInput: RandomKey ->
            val g = jit {
                    innerInput: RandomKey ->
                outerInput.floats(1) + innerInput.floats(1)
            }
            g(RandomKey(0))
        }
        val r1 = f(RandomKey(2))
        val r2 = f(RandomKey(3))
        assert(r1 !is TracingTensor && r2 !is TracingTensor)
        r1 shouldNotBe r2
    }

    @Test fun `test printing`() {
        fun f(k: RandomKey): Pair<RandomKey, DTensor> {
            val splits = k.split(5)
            val data = splits[0].floats(5)
            return Pair(splits[2], data + data)
        }

        fun f(x: DTensor): DTensor {
            return sin(x.pow(2))
        }

        val x = TracingRandomKey.Variable(0, tid)
        val y = f(x)
        val s = simplify(y)
        (s.first as TracingRandomKey).printedForm(1) shouldBe "split(key = r0, 5)[2]"
        (s.second as TracingTensor).printedForm(1) shouldBe "val t1 = split(key = r0, 5)[0].floats(Shape(5))\nt1 + t1"

    }
}
