/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import testutils.*

class LossTest: AnnotationSpec() {

    val defaultEpsilon = 1e-7f

    // TODO: Try changing this to 1e-10f (to match V1) once the max op is ported and
    //   softmax is updated. https://github.com/facebookincubator/diffkt/issues/209
    val softmaxEpsilon = 2e-7f

    @Test
    fun logSoftmaxAxis0() {
        val x = FloatTensor(Shape(2, 3), floats(6))
        val y = x.logSoftmax(axis = 0)
        val expected = FloatTensor(
            Shape(2, 3), floatArrayOf(
                -3.04858732223510740f, -3.04858732223510740f, -3.04858732223510740f,
                -0.04858732223510742f, -0.04858732223510742f, -0.04858732223510742f
        ))
        y.shouldBeNear(expected, defaultEpsilon)

        val xgrad = reverseDerivative(x) { it.logSoftmax(axis = 0).sum() }
        val expectedGrad = FloatTensor(
            Shape(2, 3), floatArrayOf(
                0.9051482677459717f, 0.9051482677459717f, 0.9051482677459717f,
                -0.9051482677459717f, -0.9051482677459717f, -0.9051482677459717f
        ))
        xgrad.shouldBeNear(expectedGrad, defaultEpsilon)
    }

    @Test
    fun logSoftmaxAxis1() {
        val x = FloatTensor(Shape(2, 3), floats(6))
        val y = x.logSoftmax(axis = 1)
        val expected = FloatTensor(
            Shape(2, 3), floatArrayOf(
                -2.40760588645935058594f, -1.40760588645935058594f, -0.40760588645935058594f,
                -2.40760612487792968750f, -1.40760612487792968750f, -0.40760612487792968750f
        ))
        y.shouldBeNear(expected, 3e-6f)

        val expectedGrad = FloatTensor(
            Shape(2, 3), floatArrayOf(
                0.7299082279205322f, 0.26581454277038574f, -0.9957230091094971f,
                0.7299083471298218f, 0.26581472158432007f, -0.9957225322723389f
        ))

        val xgrad = reverseDerivative(x) { it.logSoftmax(axis = 1).sum() }
        xgrad.shouldBeNear(expectedGrad, 5e-6f)
    }

    /**
     * TODO: implement the more-numerically-stable logSoftmax for the non-DNNL case
     *   to un-ignore this test. https://github.com/facebookincubator/diffkt/issues/211
     */
    @Test
    fun logSoftmaxNaN() {
        val x = tensorOf(
                30.033445F, 30.379223F, 31.876366F, 31.98418F, -100.50284F,
                -27.616837F, -5.4300876F, 33.250366F, -53.073635F, 26.534292F).reshape(Shape(1, 10))
        val xgrad = forwardDerivative(x) { it.logSoftmax(axis = 1) }
        assert(!xgrad.sum().value.isNaN())
    }

    @Test
    fun softmaxAxis0() {
        val x = FloatTensor(Shape(2, 3), floats(6))
        val y = x.softmax(axis = 0)
        val expected = FloatTensor(
            Shape(2, 3), floatArrayOf(
                0.04742587357759475708f, 0.04742587357759475708f, 0.04742587357759475708f,
                0.95257413387298583984f, 0.95257413387298583984f, 0.95257413387298583984f
        ))
        y.shouldBeNear(expected, softmaxEpsilon)

        val xgrad = reverseDerivative(x) { it.softmax(axis = 0).sum() }
        xgrad.shouldBeNear(FloatTensor.zeros(Shape(2, 3)), softmaxEpsilon)
    }

    @Test
    fun softmaxAxis1() {
        val x = FloatTensor(Shape(2, 3), floats(6))
        val y = x.softmax(axis = 1)
        val expected = FloatTensor(
            Shape(2, 3), floatArrayOf(
                0.09003057330846786499f, 0.24472847580909729004f, 0.66524094343185424805f,
                0.09003057330846786499f, 0.24472847580909729004f, 0.66524094343185424805f
        ))
        y.shouldBeNear(expected, softmaxEpsilon)

        val xgrad = reverseDerivative(x) { it.softmax(axis = 1).sum() }
        xgrad.shouldBeNear(FloatTensor.zeros(Shape(2, 3)), softmaxEpsilon)
    }

    @Test
    fun crossEntropyLossOneHot() {
        val t = FloatTensor(
            Shape(1, 10),
            floatArrayOf(
                -0.67616606f, 5.281276f, -4.7992377f, 0.30478132f, -1.3009043f,
                -1.6124825f, -0.8466249f, -2.3151965f, -3.4833593f, -0.41755605f
            ),
        )
        val label = FloatTensor(
            Shape(1, 10),
            floatArrayOf(0f, 0f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f)
        )
        val expectedResult = FloatScalar(7.614426f)
        val expectedGrad = FloatTensor(
            Shape(1, 10),
            floatArrayOf(0.002540f, 0.982207f, 0.000041f, 0.006775f, 0.001360f,
                0.000996f, 0.002142f, -0.999507f, 0.000153f, 0.003290f)
        )
        val (result, grad) = primalAndReverseDerivative(t) { tt ->
            crossEntropyLossFromOneHot(tt, label)
        }
        result.shouldBeNear(expectedResult, 2e-6f)
        grad.shouldBeNear(expectedGrad, 2e-6f)
    }

    @Test
    fun crossEntropyLoss() {
        val t = FloatTensor(
            Shape(1, 10),
            floatArrayOf(
                -0.67616606f, 5.281276f, -4.7992377f, 0.30478132f, -1.3009043f,
                -1.6124825f, -0.8466249f, -2.3151965f, -3.4833593f, -0.41755605f
            ),
        )
        val label = FloatTensor(Shape(1), floatArrayOf(7f))
        val expectedResult = FloatScalar(7.614426f)
        val expectedGrad = FloatTensor(
            Shape(1, 10),
            floatArrayOf(0.002540f, 0.982207f, 0.000041f, 0.006775f, 0.001360f,
                0.000996f, 0.002142f, -0.999507f, 0.000153f, 0.003290f)
        )

        val (result, grad) = primalAndReverseDerivative(t) { tt ->
            crossEntropyLoss(tt, label)
        }
        result.shouldBeNear(expectedResult, 2e-6f)
        grad.shouldBeNear(expectedGrad, 2e-6f)
    }

    @Test
    fun crossEntropyLossNaN() {
        val t = tensorOf(
                30.033445F, 30.379223F, 31.876366F, 31.98418F, -100.50284F,
                -27.616837F, -5.4300876F, 33.250366F, -53.073635F, 26.534292F).reshape(Shape(1, 10))
        val labels = tensorOf(0.0F)
        val res = crossEntropyLoss(t, labels)
        assert(!res.value.isNaN())
    }
}
