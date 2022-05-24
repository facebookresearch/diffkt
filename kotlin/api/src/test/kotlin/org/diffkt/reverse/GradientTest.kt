/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import kotlin.math.E
import kotlin.math.exp
import testutils.*

class GradientTest : AnnotationSpec() {

    fun f1(x: DTensor, y: DTensor): DScalar {
        return (x + y * exp(x)).sum()
    }

    fun f2(x: DTensor, y: DTensor): DScalar {
        return x.concat(exp(y) + y.pow(-1), 0).sum()
    }

    fun f3(x: DTensor, y: DTensor): DScalar {
        return x.matmul(y).transpose().sum()
    }

    // Consider: May be worth adding dxdy to this one in particular as testF1SecondOrder.
    @Test fun testF1() {
        val x = tensorOf(1f, 3f, 6f)
        val y = tensorOf(2f, 5f, 7f)

        val (primal1, dx) = primalAndGradient(x) { f1(it, y) }
        val (primal2, dy) = primalAndGradient(y) { f1(x, it) }

        primal1 shouldBeExactly primal2
        primal1 shouldBeExactly FloatScalar(2939.866f)

        dx shouldBeExactly tensorOf(1f + 2f*E.toFloat(), 1f + 5f*exp(3f), 1f + 7f*exp(6f))
        dy shouldBeExactly tensorOf(E.toFloat(), exp(3f), exp(6f))
    }

    @Test fun testF2() {
        val x = tensorOf(1f, 3f, 6f)
        val y = tensorOf(2f, 5f, 7f)

        val (primal, grads) = primalAndGradient(x, y, ::f2)
        val (dx, dy) = grads

        primal shouldBeExactly FloatScalar(1263.2782f)

        dx shouldBeExactly tensorOf(1f, 1f, 1f)
        dy shouldBeExactly tensorOf(exp(2f) - 0.25f, exp(5f) - 0.04f, exp(7f) - 1f/49)
    }

    @Test fun testF3() {
        val x = tensorOf(2f, 5f, 7f, 2f).reshape(2,2)
        val y = tensorOf(1f, 3f, 6f, 4f, 8f, 10f).reshape(2,3)

        val (primal, grads) = primalAndGradient(x, y, ::f3)
        val (dx, dy) = grads

        primal shouldBeExactly FloatScalar(244f)

        dx shouldBeExactly tensorOf(10f, 22f, 10f, 22f).reshape(Shape(2, 2))
        dy shouldBeExactly tensorOf(9f, 9f, 9f, 7f, 7f, 7f).reshape(Shape(2, 3))
    }
}
