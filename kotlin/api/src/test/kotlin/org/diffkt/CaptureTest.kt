/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import io.kotest.core.spec.style.AnnotationSpec
import testutils.*

class CaptureTest : AnnotationSpec() {

    private class Wrapper(var value: DTensor)

    @Test fun doubleFirstDerivativesReverse() {
        val t = tensorOf(1f,3f,5f)
        val wrapper = Wrapper(t)

        fun f1(x: DTensor) : DTensor {
            wrapper.value = x
            return 2f * x
        }

        fun f2(x: DTensor) : DTensor {
            return 5f * x
        }

        val d1 = reverseDerivative(t, ::f1) // This sets wrapper.value to a ReverseTensor
        val d2 = reverseDerivative(wrapper.value, ::f2)
        val d1Recompute = reverseDerivative(t, ::f1)

        d1 shouldBeExactly d1Recompute
        d1 shouldBeExactly tensorOf(
                2f, 0f, 0f,
                0f, 2f, 0f,
                0f, 0f, 2f
        ).reshape(3,3)
        d2 shouldBeExactly tensorOf(
                5f, 0f, 0f,
                0f, 5f, 0f,
                0f, 0f, 5f
        ).reshape(3,3)
    }

    @Test fun doubleFirstDerivativesForward() {
        val t = tensorOf(1f,3f,5f)
        val wrapper = Wrapper(t)

        fun f1(x: DTensor) : DTensor {
            wrapper.value = x
            return 2f * x
        }

        fun f2(x: DTensor) : DTensor {
            return 5f * x
        }

        val d1 = forwardDerivative(t, ::f1) // This sets wrapper.value to a ForwardTensor
        val d2 = forwardDerivative(wrapper.value, ::f2)
        val d1Recompute = forwardDerivative(t, ::f1)

        d1 shouldBeExactly d1Recompute
        d1 shouldBeExactly tensorOf(
                2f, 0f, 0f,
                0f, 2f, 0f,
                0f, 0f, 2f
        ).reshape(3,3)
        d2 shouldBeExactly tensorOf(
                5f, 0f, 0f,
                0f, 5f, 0f,
                0f, 0f, 5f
        ).reshape(3,3)
    }

    @Test fun derivativeSumReverse() {
        val t = tensorOf(1f,3f,5f)
        val wrapper = Wrapper(t)

        fun f1(x: DTensor) : DTensor {
            wrapper.value = x
            return 2f * x
        }

        fun f2(x: DTensor) : DTensor {
            return reverseDerivative(x, wrapper.value) { xx, yy -> 5f * yy + f1(xx) }.first
        }

        // Capture a ReverseTensor as wrapper.value
        reverseDerivative(t, ::f1)

        // Show that using the same ReverseTensor doesn't result in double accumulation
        val d1 = f2(wrapper.value)
        val d2 = f2(t)
        d1 shouldBeExactly d2
    }

    @Test fun derivativeSumForward() {
        val t = tensorOf(1f,3f,5f)
        val wrapper = Wrapper(t)

        fun f1(x: DTensor) : DTensor {
            wrapper.value = x
            return 2f * x
        }

        fun f2(x: DTensor) : DTensor {
            return forwardDerivative(x, wrapper.value) { xx, yy -> 5f * yy + f1(xx) }.first
        }

        // Capture a ForwardTensor as wrapper.value
        forwardDerivative(t, ::f1)

        val d1 = f2(wrapper.value)
        val d2 = f2(t)
        d1 shouldBeExactly d2
    }

}