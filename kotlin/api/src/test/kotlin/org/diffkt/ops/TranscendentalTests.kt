/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import kotlin.math.*
import testutils.*

class TranscendentalTests : AnnotationSpec() {
    val p1 = FloatScalar(0.1F)
    val p2 = FloatScalar(1.0F)
    val t1 = FloatTensor.ones(Shape(3, 2))
    val gamma = 0.577215665F

    @Test fun sinScalar() {
        sin(p1) shouldBeExactly FloatScalar(sin(0.1F))

        forwardDerivative(p1) { x: DScalar -> sin(2F*x) } shouldBeExactly 2F * cos(2F * p1)
        reverseDerivative(p1) { x: DScalar -> sin(2F*x) } shouldBeExactly 2F * cos(2F * p1)
    }

    @Test fun pi() {
        forwardDerivative(FloatScalar.PI) { x: DScalar -> sin(x) } shouldBeExactly FloatScalar(-1F)
        forwardDerivative(FloatScalar.PI / 2) { x: DScalar -> sin(x) } shouldBeExactly cos(FloatScalar.PI / 2)
    }

    @Test fun cosScalar() {
        cos(p1) shouldBeExactly FloatScalar(cos(0.1F))

        forwardDerivative(p1) { x: DScalar -> cos(2F*x) } shouldBeExactly - 2F * sin(2F * p1)
        reverseDerivative(p1) { x: DScalar -> cos(2F*x) } shouldBeExactly - 2F * sin(2F * p1)
    }

    @Test fun tanScalar() {
        tan(p1) shouldBeExactly FloatScalar(tan(0.1F))

        forwardDerivative(p1) { x: DScalar -> tan(2F*x) } shouldBeExactly 2F / (cos(2F * p1) * cos(2f * p1))
        reverseDerivative(p1) { x: DScalar -> tan(2F*x) } shouldBeExactly  2F / (cos(2F * p1) * cos(2f * p1))
    }

    @Test fun atanScalar() {
        atan(p1) shouldBeExactly FloatScalar(atan(0.1F))

        forwardDerivative(p1) { x: DScalar -> atan(2F*x) } shouldBeExactly 2F / (1f + (2F * p1).pow(2))
        reverseDerivative(p1) { x: DScalar -> atan(2F*x) } shouldBeExactly 2F / (1f + (2F * p1).pow(2))
    }

    @Test fun expScalar() {
        exp(p1) shouldBeExactly FloatScalar(exp(0.1F))

        forwardDerivative(p1) { x: DScalar -> exp(2F*x) } shouldBeExactly 2F * exp(2F * p1)
        reverseDerivative(p1) { x: DScalar -> exp(2F*x) } shouldBeExactly 2F * exp(2F * p1)
    }

    @Test fun lnScalar() {
        ln(p1) shouldBeExactly FloatScalar(ln(0.1F))

        forwardDerivative(p1) { x: DScalar -> ln(2F*x) } shouldBeExactly 1F / p1
        reverseDerivative(p1) { x: DScalar -> ln(2F*x) } shouldBeExactly 1F / p1
    }

    @Test fun lgammaScalar() {
        lgamma(p2) shouldBeExactly FloatScalar(0F)

        forwardDerivative(p2) { x: DScalar -> lgamma(2F * x) } shouldBeExactly FloatScalar(2F - 2F * gamma)
        reverseDerivative(p2) { x: DScalar -> lgamma(2F * x) } shouldBeExactly FloatScalar(2F - 2F * gamma)
    }

    @Test fun digammaScalar() {
        digamma(p2) shouldBeExactly FloatScalar(-gamma)

        forwardDerivative(p2) { x: DScalar -> digamma(2F * x) } shouldBeExactly FloatScalar((PI * PI).toFloat()/ 3F - 2F)
        reverseDerivative(p2) { x: DScalar -> digamma(2F * x) } shouldBeExactly FloatScalar((PI * PI).toFloat()/ 3F - 2F)
    }

    @Test fun polygammaScalar() {
        polygamma(1, p2) shouldBeExactly FloatScalar((PI * PI).toFloat() / 6F)
        polygamma(2, p2) shouldBeExactly FloatScalar(-2.4041138F)

        forwardDerivative(p2) { x: DScalar -> polygamma(1, 2F * x) } shouldBeExactly FloatScalar(-0.8082276F)
        reverseDerivative(p2) { x: DScalar -> polygamma(1, 2F * x) } shouldBeExactly FloatScalar(-0.8082276F)

        forwardDerivative(p2) { x: DScalar -> polygamma(2, 2F * x) } shouldBeExactly FloatScalar(2F * PI.pow(4).toFloat() / 15F - 12F)
        reverseDerivative(p2) { x: DScalar -> polygamma(2, 2F * x) } shouldBeExactly FloatScalar(2F * PI.pow(4).toFloat() / 15F - 12F)
    }

    private fun square(x: Float) = x * x
    private fun tanhDeriv(x: Float) = 1F - square(tanh(x))
    @Test fun tanhScalar() {
        tanh(p1) shouldBeExactly FloatScalar(tanh(0.1F))

        forwardDerivative(p1) { x: DScalar -> tanh(2F*x) } shouldBeExactly FloatScalar(2F * tanhDeriv(2F*p1.value))
        reverseDerivative(p1) { x: DScalar -> tanh(2F*x) } shouldBeExactly FloatScalar(2F * tanhDeriv(2F*p1.value))
    }
}
