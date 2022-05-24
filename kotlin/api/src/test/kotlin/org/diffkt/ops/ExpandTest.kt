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

class ExpandTest : AnnotationSpec() {
    @Test fun forward() {
        val x = FloatTensor(Shape(1, 1, 3)) { it.toFloat() }
        val newShape = Shape(1, 2, 3)
        fun f(x: DTensor): DTensor = x.expand(newShape)
        val der0 = f(x)
        val der1 = forwardDerivative1(x, ::f)
        val der2 = forwardDerivative2(x, ::f)
        der0 shouldBeExactly FloatTensor(newShape) { it.toFloat() % x.rank }
        der1 shouldBeExactly identityGradientofSameKind(x).expand(der0.shape + x.shape)
        der2 shouldBeExactly zeroOfSameKind(der1, der1.shape + x.shape)
    }

    @Test fun reverse() {
        val x = FloatTensor(Shape(1, 1, 3)) { it.toFloat() }
        val newShape = Shape(1, 2, 3)
        fun f(x: DTensor): DTensor = x.expand(newShape)
        val der0 = f(x)
        val der1 = reverseDerivative1(x, ::f)
        val der2 = reverseDerivative2(x, ::f)
        der0 shouldBeExactly FloatTensor(newShape) { it.toFloat() % x.rank }
        der1 shouldBeExactly identityGradientofSameKind(x).expand(x.shape + der0.shape)
        der2 shouldBeExactly zeroOfSameKind(x, x.shape + der1.shape)
    }
}
