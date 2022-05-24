/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.assertions.throwables.shouldNotThrowAny
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeExactly
import org.diffkt.*
import testutils.*
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AbsTest : AnnotationSpec() {
    @Test fun testPrimal() {
        val f = { x: DTensor -> abs(x)}
        val t1 = tensorOf(2F, -2F)
        val t2 = f(t1)
        assertTrue { t2 is FloatTensor }
        t2 shouldBeExactly tensorOf(2F, 2F).reshape(2)
    }

    @Test fun testPrimal2D() {
        val f = { x: DTensor -> abs(x)}
        val t1 = tensorOf(2F, -2F, 1F, -1F, 1F, -1F).reshape(2, 3)
        val t2 = f(t1)
        assertTrue { t2 is FloatTensor }
        t2 shouldBeExactly tensorOf(2F, 2F, 1F, 1F, 1F, 1F).reshape(2, 3)
    }

    @Test fun testForwardDerivative() {
        val f = { x: DTensor -> abs(x)}
        val t1 = tensorOf(2F, -2F)
        val d1 = forwardDerivative(t1) { x: DTensor -> f(x) }
        assertTrue { d1 is FloatTensor }
        d1 shouldBeExactly tensorOf(1F, 0F, 0F, -1F).reshape(2, 2)
    }

    @Test fun testReverseDerivative() {
        val f = { x: DTensor -> abs(x)}
        val t1 = tensorOf(2F, -2F)
        val d1 = reverseDerivative(t1) { x: DTensor -> f(x) }
        d1 shouldBeExactly tensorOf(1F, 0F, 0F, -1F).reshape(2, 2)
    }

    @Test fun testForwardDerivative2D() {
        val f = { x: DTensor -> abs(x)}
        val t1 = tensorOf(2F, -2F, 1F, -1F, 1F, -1F).reshape(2, 3)
        val d1 = forwardDerivative(t1) { x: DTensor -> f(x) }
        assertEquals(Shape(2, 3, 2, 3), d1.shape)
        assertClose(1F, d1[0, 0, 0, 0].scalarValue)
        assertClose(-1F, d1[0, 1, 0, 1].scalarValue)
        assertClose(0F, d1[0, 0, 0, 1].scalarValue)
        assertClose(1F, d1[0, 2, 0, 2].scalarValue)
    }

    @Test fun testReverseDerivative2D() {
        val f = { x: DTensor -> abs(x)}
        val t1 = tensorOf(2F, -2F, 1F, -1F, 1F, -1F).reshape(2, 3)
        val d1 = reverseDerivative(t1) { x: DTensor -> f(x) }
        assertEquals(Shape(2, 3, 2, 3), d1.shape)
        assertClose(1F, d1[0, 0, 0, 0].scalarValue)
        assertClose(-1F, d1[0, 1, 0, 1].scalarValue)
        assertClose(0F, d1[0, 0, 0, 1].scalarValue)
        assertClose(1F, d1[0, 2, 0, 2].scalarValue)
    }

    @Test fun secondOrder() {
        val f = { x: DTensor -> abs(x)}
        val t = tensorOf(2F, -2F, 1F, -1F, 1F, -1F).reshape(2, 3)
        val ff = forwardDerivative(t) { tt: DTensor -> forwardDerivative(tt) { x: DTensor -> f(x) } }
        val fr = forwardDerivative(t) { tt: DTensor -> reverseDerivative(tt) { x: DTensor -> f(x) } }
        val rf = reverseDerivative(t) { tt: DTensor -> forwardDerivative(tt) { x: DTensor -> f(x) } }
        val rr = reverseDerivative(t) { tt: DTensor -> reverseDerivative(tt) { x: DTensor -> f(x) } }

        val expected = FloatTensor.zeros(Shape(2, 3, 2, 3, 2, 3))
        ff shouldBeExactly expected
        fr shouldBeExactly expected
        rf shouldBeExactly expected
        rr shouldBeExactly expected
    }

    @Test fun `abs forward derivative shape`() {
        fun f(t: DTensor): DTensor {
            return abs(t.stack(t)).transpose()
        }
        val t = tensorOf(*floats(3))
        shouldNotThrowAny {
            forwardDerivative(t, ::f)
        }
    }

    @Test fun `abs reverse derivative shape`() {
        fun f(t: DTensor): DTensor {
            return t.stack(abs(t)).transpose()
        }
        val t = tensorOf(*floats(3))
        shouldNotThrowAny {
            reverseDerivative(t, ::f)
        }
    }

    @Test fun `abs grad at discontinuities should be one`() {
        fun f(t: DScalar): DScalar = abs(t)
        forwardDerivative(FloatScalar.ZERO, ::f).value shouldBeExactly 1f
        forwardDerivative(FloatScalar(Float.NaN), ::f).value shouldBeExactly 1f
        reverseDerivative(FloatScalar.ZERO, ::f).value shouldBeExactly 1f
        reverseDerivative(FloatScalar(Float.NaN), ::f).value shouldBeExactly 1f
    }
}