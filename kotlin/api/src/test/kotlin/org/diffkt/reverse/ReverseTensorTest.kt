/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeExactly
import org.diffkt.*
import org.diffkt.ops.elements
import testutils.*

private val DTensor.scalarValue: Float get() {
    return when (val t = this.primal(NoDerivativeID)) {
        is FloatScalar -> t.value
        else -> throw IllegalArgumentException("cannot index a ${t::class.qualifiedName}")
    }
}

class ReverseTensorTest : AnnotationSpec() {

    @Test fun testAdd() {
        val f = { x: DTensor, y: DTensor -> x + y + x }
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F, 4F)
        val t3 = f(t1, t2)
        t3 shouldBeExactly tensorOf(5F, 8F).reshape(2)
    }

    @Test fun testDerivativeAdd1() {
        val f = { x: DTensor, y: DTensor -> x + y + x }
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F, 4F)
        val d1 = reverseDerivative(t1) { x: DTensor -> f(x, t2) }
        d1 shouldBeExactly tensorOf(2F, 0F, 0F, 2F).reshape(2, 2)
        val d2 = reverseDerivative(t2) { y: DTensor -> f(t1, y) }
        d2 shouldBeExactly tensorOf(1F, 0F, 0F, 1F).reshape(2, 2)
    }

    @Test fun testDerivativeAdd2() {
        val f = { x: DTensor, y: DTensor -> x + y + x }
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F, 4F)
        val (d1, d2) = reverseDerivative(t1, t2, f)
        d1 shouldBeExactly tensorOf(2F, 0F, 0F, 2F).reshape(2, 2)
        d2 shouldBeExactly tensorOf(1F, 0F, 0F, 1F).reshape(2, 2)
    }

    @Test fun testDerivativeSub1() {
        val t1 = tensorOf(0F)
        val t2 = tensorOf(1F)
        val d2 = reverseDerivative(t2) { x: DTensor -> t1 - x }
        assertClose(-1F, d2[0, 0].scalarValue)
    }

    @Test fun testDerivativeSub2() {
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F, 4F)
        val f = { x: DTensor, y: DTensor -> (- x) - y - x }

        val d1 = reverseDerivative(t1) { x: DTensor -> f(x, t2) }
        d1 shouldBeExactly tensorOf(-2F, 0F, 0F, -2F).reshape(2, 2)

        val d2 = reverseDerivative(t2) { y: DTensor -> f(t1, y) }
        d2 shouldBeExactly tensorOf(-1F, 0F, 0F, -1F).reshape(2, 2)
    }

    @Test fun testAggregate_1_1() {
        fun f1(x: DScalar): DScalar = 10F * x * x * x

        fun f(t: DTensor): DTensor {
            assert(t.shape == Shape(1))
            val x = t.elements[0]
            return tensorOf(f1(x))
        }

        val x = FloatScalar(100F)
        val input = tensorOf(x)

        // Test the non-derivative case
        val res = f(input)
        assert(res.shape == Shape(1))
        for (i in 0 until res.shape.first) {
            assertClose(f1(x).value, res[i].scalarValue)
        }

        // First derivative
        val der = reverseDerivative1(input, ::f)
        assert(der.shape == Shape(1, 1))
        assertClose(reverseDerivative(x) { xx: DScalar -> f1(xx) }.value, der[0, 0].scalarValue)

        // Second derivative
        val der2 = reverseDerivative2(input, ::f)
        assert(der2.shape == Shape(1, 1, 1))
        assertClose(reverseDerivative(x) { x1: DScalar -> reverseDerivative(x1) { x2: DScalar -> f1(x2) } }.value, der2[0, 0, 0].scalarValue)
    }

    @Test fun testAggregate_1_2() {
        val nin = 1
        val nout: Int = 2

        fun f0(x: DScalar): DScalar = x * x
        fun f1(x: DScalar): DScalar = x * x * x

        fun f(t: DTensor): DTensor {
            assert(t.shape == Shape(nin))
            val x = t.elements[0]
            return tensorOf(f0(x), f1(x))
        }

        val x = FloatScalar(100F)
        val input = tensorOf(x)

        // Test the non-derivative case
        val res = f(input)
        res shouldBeExactly tensorOf(f0(x).value, f1(x).value).reshape(nout)

        // First derivative
        val der = reverseDerivative1(input, ::f)
        der shouldBeExactly tensorOf(reverseDerivative(x, ::f0).value, reverseDerivative(x, ::f1).value).reshape(nin, nout)

        // Second derivative
        val der2 = reverseDerivative2(input, ::f)
        der2 shouldBeExactly tensorOf(
                reverseDerivative(x) { x1: DScalar -> reverseDerivative(x1, ::f0) }.value,
                reverseDerivative(x) { x1: DScalar -> reverseDerivative(x1, ::f1) }.value
            ).reshape(nin, nin, nout)
    }

    @Test fun testAggregate_2_1() {
        val nin: Int = 2
        val nout: Int = 1

        fun f0(x: DScalar, y: DScalar): DScalar = (x + 3F) * (y + 5F)

        fun f(t: DTensor): DTensor {
            assert(t.shape == Shape(nin))
            val tElements = t.elements
            val x = tElements[0]
            val y = tElements[1]
            return tensorOf(f0(x, y))
        }

        val x = FloatScalar(17F)
        val y = FloatScalar(23F)
        val input = tensorOf(x, y)

        // Test the non-derivative case
        val res = f(input)
        assert(res.shape == Shape(nout))
        assertClose(f0(x, y).value, res[0].scalarValue)

        // First derivative
        run {
            val der = reverseDerivative1(input, ::f)
            assert(der.shape == Shape(nin, nout))
            assertClose(reverseDerivative(x) { x1: DScalar -> f0(x1, y) }.value, der[0, 0].scalarValue)
            assertClose(reverseDerivative(y) { y1: DScalar -> f0(x, y1) }.value, der[1, 0].scalarValue)
        }

        // Second derivative
        run {
            val der2 = reverseDerivative2(input, ::f)
            assert(der2.shape == Shape(nin, nin, nout))
            assertClose(der2[1, 0, 0].scalarValue, der2[0, 1, 0].scalarValue) // the hessian is symmetric
            assertClose(reverseDerivative(x) { x1: DScalar -> reverseDerivative(x1) { x2: DScalar -> f0(x2, y) } }.value, der2[0, 0, 0].scalarValue)
            assertClose(reverseDerivative(x) { x1: DScalar -> reverseDerivative(y) { y2: DScalar -> f0(x1, y2) } }.value, der2[0, 1, 0].scalarValue)
            assertClose(reverseDerivative(y) { y1: DScalar -> reverseDerivative(x) { x2: DScalar -> f0(x2, y1) } }.value, der2[1, 0, 0].scalarValue)
            assertClose(reverseDerivative(y) { y1: DScalar -> reverseDerivative(y1) { y2: DScalar -> f0(x, y2) } }.value, der2[1, 1, 0].scalarValue)
        }
    }

    @Test fun testAggregate_3_8() {
        val nin: Int = 3 // x, y, and z

        val functions: Array<(x: DScalar, y: DScalar, z: DScalar) -> DScalar> = arrayOf(
                { _: DScalar, _: DScalar, _: DScalar -> FloatScalar(5F) },
                { x: DScalar, y: DScalar, z: DScalar -> 3F * x * x + 4F * x * y + 5F * y * z },
                { x: DScalar, y: DScalar, z: DScalar -> 3F * sin(x + y) + 4F * cos(x + z) + 5F * tan(y + z) },
                { x: DScalar, y: DScalar, z: DScalar -> (x + 3F) * (y + 4F) * (z + 5F) },
                { x: DScalar, y: DScalar, z: DScalar -> sqrt((x + 2F) * (y + 3F) * (z + 4F)) },
                { x: DScalar, _: DScalar, _: DScalar -> x * x },
                { _: DScalar, y: DScalar, _: DScalar -> y * y },
                { _: DScalar, _: DScalar, z: DScalar -> z * z }
        )

        val nout: Int = functions.size

        fun f(t: DTensor): DTensor {
            assert(t.shape == Shape(nin))
            val inputs = t.elements
            val x = inputs[0]
            val y = inputs[1]
            val z = inputs[2]
            return tensorOf(functions.map { it.invoke(x, y, z) })
        }

        val x = FloatScalar(11F)
        val y = FloatScalar(13F)
        val z = FloatScalar(17F)
        val input = tensorOf(x, y, z)

        // Test the non-derivative case
        val res = f(input)
        assert(res.shape == Shape(nout))
        for (i in 0 until nout) {
            assertClose(functions[i](x, y, z).value, res[i].scalarValue)
        }

        // Test the first derivative
        val der = reverseDerivative1(input, ::f)
        assert(der.shape == Shape(nin, nout))
        fun d1(i: Int, j: Int) = der[i, j].scalarValue
        for (i in 0 until nout) {
            val f = functions[i]
            assertClose(reverseDerivative(x) { xx: DScalar -> f(xx, y, z) }.value, d1(0, i))
            assertClose(reverseDerivative(y) { yy: DScalar -> f(x, yy, z) }.value, d1(1, i))
            assertClose(reverseDerivative(z) { zz: DScalar -> f(x, y, zz) }.value, d1(2, i))
        }

        // Test the second derivative
        val der2 = reverseDerivative2(input, ::f)
        assert(der2.shape == Shape(nin, nin, nout))
        fun d2(i: Int, j: Int, k: Int) = der2[i, j, k].scalarValue
        for (i in 0 until nout) {
            // The hessian is symmetric
            assertClose(d2(0,1,i), d2(1,0,i))
            assertClose(d2(0,2,i), d2(2,0,i))
            assertClose(d2(1,2,i), d2(2,1,i))

            val f = functions[i]
            assertClose(reverseDerivative(x) { x1: DScalar -> reverseDerivative(x1) { x2: DScalar -> f(x2, y, z) } }.value, d2(0,0,i))
            assertClose(reverseDerivative(x) { x1: DScalar -> reverseDerivative(y) { y2: DScalar -> f(x1, y2, z) } }.value, d2(0,1,i))
            assertClose(reverseDerivative(x) { x1: DScalar -> reverseDerivative(z) { z2: DScalar -> f(x1, y, z2) } }.value, d2(0,2,i))
            assertClose(reverseDerivative(y) { y1: DScalar -> reverseDerivative(x) { x2: DScalar -> f(x2, y1, z) } }.value, d2(1,0,i))
            assertClose(reverseDerivative(y) { y1: DScalar -> reverseDerivative(y1) { y2: DScalar -> f(x, y2, z) } }.value, d2(1,1,i))
            assertClose(reverseDerivative(y) { y1: DScalar -> reverseDerivative(z) { z2: DScalar -> f(x, y1, z2) } }.value, d2(1,2,i))
            assertClose(reverseDerivative(z) { z1: DScalar -> reverseDerivative(x) { x2: DScalar -> f(x2, y, z1) } }.value, d2(2,0,i))
            assertClose(reverseDerivative(z) { z1: DScalar -> reverseDerivative(y) { y2: DScalar -> f(x, y2, z1) } }.value, d2(2,1,i))
            assertClose(reverseDerivative(z) { z1: DScalar -> reverseDerivative(z1) { z2: DScalar -> f(x, y, z2) } }.value, d2(2,2,i))
        }

        der2.shouldBeNear(
            tensorOf(
            0.0F, 6.0F, 6.5671587F, 0.0F, -0.09776754F, 2.0F, 0.0F, 0.0F,
            0.0F, 4.0F, 2.7167351F, 22.0F, 0.0794361F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, 3.8504236F, 17.0F, 0.06052275f, 0.0F, 0.0F, 0.0F,

            0.0F, 4.0F, 2.7167351F, 22.0F, 0.07943611F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, -2689.3328F, 0.0F, -0.064541854F, 0.0F, 2.0F, 0.0F,
            0.0F, 5.0F, -2692.0496F, 14.0F, 0.049174733f, 0.0F, 0.0F, 0.0F,

            0.0F, 0.0F, 3.8504236F, 17.0F, 0.06052275f, 0.0F, 0.0F, 0.0F,
            0.0F, 5.0F, -2692.0496F, 14.0F, 0.049174733F, 0.0F, 0.0F, 0.0F,
            0.0F, 0.0F, -2688.1992F, 0.0F, -0.037466474f, 0.0F, 0.0F, 2.0F)
            .reshape(3, 3, 8), 1e-8F)
    }

    @Test fun testVjp() {
        fun f(x: DTensor) = x.pow(2).flatten()
        val x = tensorOf(1f, 2f, 3f, 4f, 5f, 6f).reshape(Shape(3, 2))
        val vjp = vjp(x, FloatTensor.ones(Shape(6)), ::f)
        assert(vjp.shape == Shape(3, 2))
        vjp shouldBeExactly tensorOf(2.0F, 4.0F, 6.0F, 8.0F, 10.0F, 12.0F).reshape(Shape(3, 2))
    }

    @Test fun testPrimalAndPullback() {
        fun f(x: DTensor): DTensor = x.pow(2).sum()
        val a: DTensor = FloatScalar(3f)
        val (b, pullback) = primalAndPullback(a, ::f)
        val db = FloatScalar(1f)
        val da = pullback(db)
        (da.basePrimal() as FloatScalar).value shouldBeExactly 6f
    }
}
