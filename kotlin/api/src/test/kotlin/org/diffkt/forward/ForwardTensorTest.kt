/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.forward

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import org.diffkt.ops.elements
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import testutils.*

private val DTensor.scalarValue: Float get() {
    return when (val t = this.primal(NoDerivativeID)) {
        is FloatScalar -> t.value
        else -> throw IllegalArgumentException("cannot index a ${t::class.qualifiedName}")
    }
}

class ForwardTensorTest : AnnotationSpec() {

    @Test fun testTimes() {
        val f = { x: DTensor, y: DTensor -> x * y * x }
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F, 4F)
        val t3 = f(t1, t2)
        assertTrue(t3 is FloatTensor)
        assertEquals(Shape(2), t3.shape)
        assertClose(3F, t3[0].scalarValue)
        assertClose(16F, t3[1].scalarValue)
    }

    @Test fun testDerivativeTimes() {
        val f = { x: DTensor, y: DTensor -> x * y * x }
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F, 4f)

        val d1 = forwardDerivative1(t1) { x: DTensor -> f(x, t2) }
        val d2 = forwardDerivative1(t2) { y: DTensor -> f(t1, y) }

        assertEquals(Shape(2, 2), d1.shape)
        assertClose(6F, d1[0, 0].scalarValue)
        assertClose(16F, d1[1, 1].scalarValue)

        assertEquals(Shape(2,2), d2.shape)
        assertClose(1F, d2[0, 0].scalarValue)
        assertClose(4F, d2[1, 1].scalarValue)
    }

    @Test fun testDerivativeTimesBroadcasted() {
        val f = { x: DTensor, y: DTensor -> x * y * x }
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F)

        val d1 = forwardDerivative1(t1) { x: DTensor -> f(x, t2) }
        val d2 = forwardDerivative1(t2) { y: DTensor -> f(t1, y) }

        assertEquals(Shape(2, 2), d1.shape)
        assertClose(6F, d1[0, 0].scalarValue)
        assertClose(12F, d1[1, 1].scalarValue)

        assertEquals(Shape(2, 1), d2.shape)
        assertClose(1F, d2[0, 0].scalarValue)
        assertClose(4F, d2[1, 0].scalarValue)
    }

    @Test fun testAdd() {
        val f = { x: DTensor, y: DTensor -> x + y + x }
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F, 4F)
        val t3 = f(t1, t2)
        assertTrue(t3 is FloatTensor)
        assertEquals(Shape(2), t3.shape)
        assertClose(5F, t3[0].scalarValue)
        assertClose(8F, t3[1].scalarValue)
    }

    @Test fun testDerivativeAdd1() {
        val f = { x: DTensor, y: DTensor -> x + y + x }
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F, 4F)
        val d1 = forwardDerivative(t1) { x: DTensor -> f(x, t2) }
        d1 shouldBeExactly tensorOf(2F, 0F, 0F, 2F).reshape(2, 2)
        val d2 = forwardDerivative(t2) { y: DTensor -> f(t1, y) }
        d2 shouldBeExactly tensorOf(1F, 0F, 0F, 1F).reshape(2, 2)
    }

    @Test fun testDerivativeAdd2() {
        val f = { x: DTensor, y: DTensor -> x + y + x }
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F, 4F)
        val (d1, d2) = forwardDerivative(t1, t2, f)
        d1 shouldBeExactly tensorOf(2F, 0F, 0F, 2F).reshape(2, 2)
        d2 shouldBeExactly tensorOf(1F, 0F, 0F, 1F).reshape(2, 2)
    }

    @Test fun testAddTensorScalar() {
        val x = tensorOf(1f, 2f)
        val y = 2f + x + 3f
        y shouldBeExactly tensorOf(6f, 7f)
    }

    @Test fun testDerivativeAddTensorScalar() {
        val t1 = tensorOf(1f, 2f)
        val d1 = forwardDerivative(t1) { x: DTensor -> x + 9f }
        d1 shouldBeExactly tensorOf(1f, 0f, 0f, 1f).reshape(2, 2)
    }

    @Test fun testDerivativeSub1() {
        val t1 = tensorOf(0F)
        val t2 = tensorOf(1F)
        val d2 = forwardDerivative(t2) { x: DTensor -> t1 - x }
        assertEquals(Shape(1, 1), d2.shape)
        assertClose(-1F, d2[0, 0].scalarValue)
    }

    @Test fun testDerivativeSub2a() {
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F, 4F)
        val f = { x: DTensor, y: DTensor -> (- x) - y - x }

        val d1 = forwardDerivative(t1) { x: DTensor -> f(x, t2) }
        d1 shouldBeExactly tensorOf(-2F, 0F, 0F, -2F).reshape(2, 2)

        val d2 = forwardDerivative(t2) { y: DTensor -> f(t1, y) }
        d2 shouldBeExactly tensorOf(-1F, 0F, 0F, -1F).reshape(2, 2)
    }

    @Test fun testDerivativeSub2b() {
        val t1 = tensorOf(1F, 2F)
        val t2 = tensorOf(3F, 4F)
        val f = { x: DTensor, y: DTensor -> (- x) - y - x }
        val (d1, d2) = forwardDerivative(t1, t2, f)
        d1 shouldBeExactly tensorOf(-2F, 0F, 0F, -2F).reshape(2, 2)
        d2 shouldBeExactly tensorOf(-1F, 0F, 0F, -1F).reshape(2, 2)
    }

    @Test fun testDerivativeSubTensorScalar() {
        val t1 = tensorOf(1f, 2f)
        val d1 = forwardDerivative(t1) { x: DTensor -> x - 9f }
        d1 shouldBeExactly tensorOf(1f, 0f, 0f, 1f).reshape(2, 2)
    }

    @Test fun testDerivativeSubScalarTensor() {
        val t1 = tensorOf(1f, 2f)
        val d1 = forwardDerivative(t1) { x: DTensor -> 1f - x }
        d1 shouldBeExactly tensorOf(-1f, 0f, 0f, -1f).reshape(2, 2)
    }

    @Test fun testDerivativeSubBroadcast() {
        val t1 = tensorOf(3f, 2f)
        val t2 = tensorOf(4f, 5f, 1f, 3f).reshape(2,2)
        t1 - t2 shouldBeExactly tensorOf(-1f, -3f, 2f, -1f).reshape(2, 2)
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
        val der = forwardDerivative1(input, ::f)
        assert(der.shape == Shape(1, 1))
        assertClose(forwardDerivative(x) { xx: DScalar -> f1(xx) }.value, der[0, 0].scalarValue)

        // Second derivative
        val der2 = forwardDerivative2(input, ::f)
        assert(der2.shape == Shape(1, 1, 1))
        assertClose(forwardDerivative(x) { x1: DScalar -> forwardDerivative(x1) { x2: DScalar -> f1(x2) } }.value, der2[0, 0, 0].scalarValue)
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
        assert(res.shape == Shape(nout))
        assertClose(f0(x).value, res[0].scalarValue)
        assertClose(f1(x).value, res[1].scalarValue)

        // First derivative
        val der = forwardDerivative1(input, ::f)
        assert(der.shape == Shape(nout, nin))
        assertClose(forwardDerivative(x, ::f0).value, der[0, 0].scalarValue)
        assertClose(forwardDerivative(x, ::f1).value, der[1, 0].scalarValue)

        // Second derivative
        val der2 = forwardDerivative2(input, ::f)
        assert(der2.shape == Shape(nout, nin, nin))
        assertClose(forwardDerivative(x) { x1: DScalar -> forwardDerivative(x1, ::f0) }.value, der2[0, 0, 0].scalarValue)
        assertClose(forwardDerivative(x) { x1: DScalar -> forwardDerivative(x1, ::f1) }.value, der2[1, 0, 0].scalarValue)
    }

    @Test fun testAggregate_2_1() {
        val nin: Int = 2
        val nout: Int = 1

        fun f0(x: DScalar, y: DScalar): DScalar = x * y

        fun f(t: DTensor): DTensor {
            assert(t.shape == Shape(nin))
            val tv = t.elements
            val x = tv[0]
            val y = tv[1]
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
            val der = forwardDerivative1(input, ::f)
            assert(der.shape == Shape(nout, nin))
            assertClose(forwardDerivative(x) { x1: DScalar -> f0(x1, y) }.value, der[0, 0].scalarValue)
            assertClose(forwardDerivative(y) { y1: DScalar -> f0(x, y1) }.value, der[0, 1].scalarValue)
        }

        // Second derivative
        run {
            val der2 = forwardDerivative2(input, ::f)
            assert(der2.shape == Shape(nout, nin, nin))
            assertClose(der2[0, 0, 1].scalarValue, der2[0, 1, 0].scalarValue) // the hessian is symmetric
            assertClose(forwardDerivative(x) { x1: DScalar -> forwardDerivative(x1) { x2: DScalar -> f0(x2, y) } }.value, der2[0, 0, 0].scalarValue)
            assertClose(forwardDerivative(x) { x1: DScalar -> forwardDerivative(y) { y2: DScalar -> f0(x1, y2) } }.value, der2[0, 0, 1].scalarValue)
            assertClose(forwardDerivative(y) { y1: DScalar -> forwardDerivative(x) { x2: DScalar -> f0(x2, y1) } }.value, der2[0, 1, 0].scalarValue)
            assertClose(forwardDerivative(y) { y1: DScalar -> forwardDerivative(y1) { y2: DScalar -> f0(x, y2) } }.value, der2[0, 1, 1].scalarValue)
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
            val tv = t.elements
            val x = tv[0]
            val y = tv[1]
            val z = tv[2]
            return tensorOf(functions.map { it.invoke(x, y, z) })
        }

        val x = FloatScalar(11F)
        val y = FloatScalar(13F)
        val z = FloatScalar(17F)
        val input = tensorOf(x, y, z)

        // Test the non-derivative case
        val res = f(input)
        assert(res.shape == Shape(functions.size))
        for (i in 0 until nout) {
            assertClose(functions[i](x, y, z).value, res[i].scalarValue)
        }

        // Test the first derivative
        val der = forwardDerivative1(input, ::f)
        assert(der.shape == Shape(functions.size, nin))
        for (i in 0 until nout) {
            val f = functions[i]
            assertClose(forwardDerivative(x) { xx: DScalar -> f(xx, y, z) }.value, der[i, 0].scalarValue)
            assertClose(forwardDerivative(y) { yy: DScalar -> f(x, yy, z) }.value, der[i, 1].scalarValue)
            assertClose(forwardDerivative(z) { zz: DScalar -> f(x, y, zz) }.value, der[i, 2].scalarValue)
        }

        // Test the second derivative
        val der2 = forwardDerivative2(input, ::f)
        assert(der2.shape == Shape(nout, nin, nin)) // 8,3,3
        fun d2(i: Int, j: Int, k: Int) = der2[i, j, k].scalarValue
        for (i in 0 until nout) {
            // The hessian is symmetric
            assertClose(d2(i, 0, 1), d2(i, 1, 0))
            assertClose(d2(i, 0, 2), d2(i, 2, 0))
            assertClose(d2(i, 2, 1), d2(i, 1, 2))

            val f = functions[i]
            assertClose(forwardDerivative(x) { x1: DScalar -> forwardDerivative(x1) { x2: DScalar -> f(x2, y, z) } }.value, d2(i, 0, 0))
            assertClose(forwardDerivative(x) { x1: DScalar -> forwardDerivative(y) { y2: DScalar -> f(x1, y2, z) } }.value, d2(i, 0, 1))
            assertClose(forwardDerivative(x) { x1: DScalar -> forwardDerivative(z) { z2: DScalar -> f(x1, y, z2) } }.value, d2(i, 0, 2))
            assertClose(forwardDerivative(y) { y1: DScalar -> forwardDerivative(x) { x2: DScalar -> f(x2, y1, z) } }.value, d2(i, 1, 0))
            assertClose(forwardDerivative(y) { y1: DScalar -> forwardDerivative(y1) { y2: DScalar -> f(x, y2, z) } }.value, d2(i, 1, 1))
            assertClose(forwardDerivative(y) { y1: DScalar -> forwardDerivative(z) { z2: DScalar -> f(x, y1, z2) } }.value, d2(i, 1, 2))
            assertClose(forwardDerivative(z) { z1: DScalar -> forwardDerivative(x) { x2: DScalar -> f(x2, y, z1) } }.value, d2(i, 2, 0))
            assertClose(forwardDerivative(z) { z1: DScalar -> forwardDerivative(y) { y2: DScalar -> f(x, y2, z1) } }.value, d2(i, 2, 1))
            assertClose(forwardDerivative(z) { z1: DScalar -> forwardDerivative(z1) { z2: DScalar -> f(x, y, z2) } }.value, d2(i, 2, 2))
        }
    }

    @Test fun testJvp() {
        fun f(x: DTensor) = x.pow(2).flatten()
        val x = tensorOf(1f, 2f, 3f, 4f, 5f, 6f).reshape(Shape(3, 2))
        val jvp = jvp(x, FloatTensor.ones(x.shape), ::f)
        assert(jvp.shape == Shape(6))
        jvp shouldBeExactly tensorOf(2.0F, 4.0F, 6.0F, 8.0F, 10.0F, 12.0F)
    }
}
