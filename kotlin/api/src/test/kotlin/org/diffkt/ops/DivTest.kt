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

class DivTest : AnnotationSpec() {

    @Test fun divTest1() {
        val t1 = tensorOf(3f, 4f, 5f, 6f).reshape(2,2)
        val t2 = tensorOf(3f, 2f, 2f, 4f).reshape(2,2)
        t1/t2 shouldBeExactly tensorOf(1f, 2f, 2.5f, 1.5f).reshape(2,2)
    }

    @Test fun divTestForward() {
        val t1 = tensorOf(3f, 4f, 5f, 6f).reshape(2,2)
        val t2 = tensorOf(3f, 2f, 2f, 4f).reshape(2,2)
        val d1 = forwardDerivative(t1) { t -> t/t2 }
        val d2 = forwardDerivative(t2) { t -> t1/t }
        d1 shouldBeExactly tensorOf(
                1/3f, 0f, 0f, 0f,
                0f, 0.5f, 0f, 0f,
                0f, 0f, 0.5f, 0f,
                0f, 0f, 0f, 0.25f
        ).reshape(2, 2, 2, 2)
        d2 shouldBeExactly tensorOf(
                -1/3f, 0f, 0f, 0f,
                0f, -1f, 0f, 0f,
                0f, 0f, -1.25f, 0f,
                0f, 0f, 0f, -3/8f
        ).reshape(2,2,2,2)

        val dd1 = forwardDerivative2(t1) { t -> t/t2 }
        val dd2 = forwardDerivative2(t2) { t -> t1/t }
        dd1 shouldBeExactly FloatTensor.zeros(Shape(2, 2, 2, 2, 2, 2))
        dd2 shouldBeExactly tensorOf(
                2/9f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1.25f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0.1875f,
        ).reshape(2,2,2,2,2,2)
    }

    @Test fun divTestReverse() {
        val t1 = tensorOf(3f, 4f, 5f, 6f).reshape(2,2)
        val t2 = tensorOf(3f, 2f, 2f, 4f).reshape(2,2)
        val d1 = reverseDerivative(t1) { t -> t/t2 }
        val d2 = reverseDerivative(t2) { t -> t1/t }
        d1 shouldBeExactly tensorOf(
                1/3f, 0f, 0f, 0f,
                0f, 0.5f, 0f, 0f,
                0f, 0f, 0.5f, 0f,
                0f, 0f, 0f, 0.25f
        ).reshape(2, 2, 2, 2)
        d2 shouldBeExactly tensorOf(
                -1/3f, 0f, 0f, 0f,
                0f, -1f, 0f, 0f,
                0f, 0f, -1.25f, 0f,
                0f, 0f, 0f, -3/8f
        ).reshape(2,2,2,2)
        val dd1 = reverseDerivative2(t1) { t -> t/t2 }
        val dd2 = reverseDerivative2(t2) { t -> t1/t }
        dd1 shouldBeExactly FloatTensor.zeros(Shape(2, 2, 2, 2, 2, 2))
        dd2 shouldBeExactly tensorOf(
                2/9f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1.25f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0.1875f,
        ).reshape(2,2,2,2,2,2)
    }

    @Test fun divBroadcastTest() {
        val t1 = tensorOf(3f, 4f, 8f, 6f).reshape(2,2)
        val t2 = tensorOf(3f)
        t1/t2 shouldBeExactly tensorOf(1f, 4/3f, 8/3f, 2f).reshape(2,2)
    }

    @Test fun divBroadcastTestForward() {
        val t1 = tensorOf(3f, 4f, 8f, 6f).reshape(2,2)
        val t2 = tensorOf(3f)

        val d1 = forwardDerivative(t1) { t -> t/t2 }
        val d2 = forwardDerivative(t2) { t -> t1/t }
        d1 shouldBeExactly tensorOf(
                1/3f, 0f, 0f, 0f,
                0f, 1/3f, 0f, 0f,
                0f, 0f, 1/3f, 0f,
                0f, 0f, 0f, 1/3f
        ).reshape(2,2,2,2)
        d2 shouldBeExactly tensorOf(-1/3f, -4/9f, -8/9f, -2/3f).reshape(2,2,1)

        val dd1 = forwardDerivative2(t1) { t -> t/t2 }
        val dd2 = forwardDerivative2(t2) { t -> t1/t }
        dd1 shouldBeExactly FloatTensor.zeros(Shape(2, 2, 2, 2, 2, 2))
        dd2 shouldBeExactly tensorOf(2/9f, 8/27f, 16/27f, 12/27f).reshape(2,2,1,1)
    }

    @Test fun divBroadcastTestReverse() {
        val t1 = tensorOf(3f, 4f, 8f, 6f).reshape(2,2)
        val t2 = tensorOf(3f)

        val d1 = reverseDerivative(t1) { t -> t/t2 }
        val d2 = reverseDerivative(t2) { t -> t1/t }
        d1 shouldBeExactly tensorOf(
                1/3f, 0f, 0f, 0f,
                0f, 1/3f, 0f, 0f,
                0f, 0f, 1/3f, 0f,
                0f, 0f, 0f, 1/3f
        ).reshape(2,2,2,2)
        d2 shouldBeExactly tensorOf(-1/3f, -4/9f, -8/9f, -2/3f).reshape(1,2,2)

        val dd1 = reverseDerivative2(t1) { t -> t/t2 }
        val dd2 = reverseDerivative2(t2) { t -> t1/t }
        dd1 shouldBeExactly FloatTensor.zeros(Shape(2, 2, 2, 2, 2, 2))
        dd2 shouldBeExactly tensorOf(2/9f, 8/27f, 16/27f, 12/27f).reshape(1,1,2,2)
    }
}
