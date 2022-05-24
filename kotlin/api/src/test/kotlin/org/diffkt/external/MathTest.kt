/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.external

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe

class MathTest : AnnotationSpec() {
    @Test fun timesTest() {
        val a1 = floatArrayOf(2f, 0f, -2f, 3f, 4f)
        val a2 = floatArrayOf(9f, 11.2f, -21f, -2f, 2f)
        Math.times(a1, a2, 5) shouldBe floatArrayOf(18f, 0f, 42f, -6f, 8f)
    }

    @Test fun plusTest() {
        val a1 = floatArrayOf(2f, 0f, -2f, 3f, 4f)
        val a2 = floatArrayOf(9f, 11.2f, -21f, -2f, 2f)
        Math.plus(a1, a2, 5) shouldBe floatArrayOf(11f, 11.2f, -23f, 1f, 6f)
    }

    @Test fun minusTest() {
        val a1 = floatArrayOf(2f, 0f, -2f, 3f, 4f)
        val a2 = floatArrayOf(9f, 11.2f, -21f, -2f, 2f)
        Math.minus(a1, a2, 5) shouldBe floatArrayOf(-7f, -11.2f, 19f, 5f, 2f)
    }

    @Test fun unaryMinusTest() {
        val a = floatArrayOf(9f, 11.2f, -21f, -2f, 2f)
        Math.unaryMinus(a, 5) shouldBe floatArrayOf(-9f, -11.2f, 21f, 2f, -2f)
    }

    @Test fun expTest() {
        val a = floatArrayOf(9f, 11.2f, -21f, -2f, 2f)
        Math.exp(a, 5) shouldBe floatArrayOf(8103.084f, 73130.43f, 7.5825607E-10f, 0.13533528f, 7.389056f)
    }

    @Test fun logTest() {
        val a = floatArrayOf(9f, 11.2f, 21f, 0.2f, 1f)
        Math.log(a, 5) shouldBe floatArrayOf(2.1972246f, 2.4159138f, 3.0445225f, -1.609438f, 0.0f)
    }

    @Test fun lgammaTest() {
        val a = floatArrayOf(9f, 11.2f, 21f, 0.2f, 1f)
        Math.lgamma(a) shouldBe floatArrayOf(10.604603f, 15.576654f, 42.335617f, 1.5240638f, 0.0f)
    }

    @Test fun digammaTest() {
        val a = floatArrayOf(9f, 11.2f, 21f, 0.2f, 1f)
        Math.digamma(a) shouldBe floatArrayOf(2.1406415f, 2.3706071f, 3.020524f, -5.2890396f, -0.5772157f)
    }

    @Test fun polygammaTest() {
        val a = floatArrayOf(9f, 11.2f, 21f, 0.2f, 1f)
        Math.polygamma(1, a) shouldBe floatArrayOf(0.11751202f, 0.09339013f, 0.048770823f, 26.267376f, 1.644934f)
        Math.polygamma(2, a) shouldBe floatArrayOf(-0.013793319f, -0.008715412f, -0.0023781224f, -251.47803f, -2.4041138f)
    }
}
