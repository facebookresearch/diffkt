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

class SqueezeTest : AnnotationSpec() {
    @Test fun forward() {
        val x = FloatTensor(Shape(2, 1)) { it.toFloat() }
        fun f(x: DTensor): DTensor = x.squeeze(1)
        val der0 = f(x)
        val der1 = forwardDerivative1(x, ::f)
        val der2 = forwardDerivative2(x, ::f)
        der0 shouldBeExactly FloatTensor(Shape(2), floatArrayOf(0f, 1f))
        der1 shouldBeExactly FloatTensor(Shape(2, 2, 1), floatArrayOf(1f, 0f, 0f, 1f))
        der2 shouldBeExactly FloatTensor(Shape(2, 2, 1, 2, 1)) { 0f }
    }

    @Test fun reverse() {
        val x = FloatTensor(Shape(2, 1)) { it.toFloat() }
        fun f(x: DTensor): DTensor = x.squeeze(1)
        val der0 = f(x)
        val der1 = reverseDerivative1(x, ::f)
        val der2 = reverseDerivative2(x, ::f)
        der0 shouldBeExactly FloatTensor(Shape(2), floatArrayOf(0f, 1f))
        der1 shouldBeExactly FloatTensor(Shape(2, 1, 2), floatArrayOf(1f, 0f, 0f, 1f))
        der2 shouldBeExactly FloatTensor(Shape(2, 1, 2, 1, 2)) { 0f }
    }
}
