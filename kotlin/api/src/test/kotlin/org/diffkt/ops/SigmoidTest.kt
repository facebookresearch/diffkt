/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import kotlin.math.exp
import testutils.*

class SigmoidTest : AnnotationSpec() {

    @Test fun testSigmoid() {
        val f = { x: DTensor -> sigmoid(x) }
        val t = tensorOf(2F, -3F)
        f(t) shouldBeExactly tensorOf(1f/(1f + exp(-2f)), 1f/(1f + exp(3f)))
    }

    @Test fun testSigmoidForwardDerivatives() {
        val f = { x: DTensor -> sigmoid(x) }
        val t = tensorOf(2F, -3F)
        val d1 = forwardDerivative(t, f)
        val d2 = forwardDerivative2(t, f)
        d1 shouldBeExactly tensorOf(
                0.10499363F, 0.0F, 0.0F, 0.04517666F
        ).reshape(Shape(2, 2))
        d2 shouldBeExactly tensorOf(
                -0.07996252F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.040891577F
        ).reshape(Shape(2, 2, 2))
    }

    @Test fun testSigmoidReverseDerivatives() {
        val f = { x: DTensor -> sigmoid(x) }
        val t = tensorOf(2F, -3F)
        val d1 = reverseDerivative(t, f)
        val d2 = reverseDerivative2(t, f)
        d1 shouldBeExactly tensorOf(
                0.10499363F, 0.0F, 0.0F, 0.04517666F
        ).reshape(Shape(2, 2))
        d2 shouldBeExactly tensorOf(
                -0.07996252F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.040891573F
        ).reshape(Shape(2, 2, 2))
    }
}
