/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe
import org.diffkt.*

class FlattenTest: AnnotationSpec() {

    fun f(x: DTensor): DTensor = x.flatten()

    @Test fun forward() {
        val x = FloatTensor(Shape(2, 3)) { it.toFloat() }
        val der0 = f(x)
        val der1 = forwardDerivative1(x, ::f)
        val der2 = forwardDerivative2(x, ::f)
        der0.shape shouldBe Shape(6)
        der1.shape shouldBe Shape(6, 2, 3)
        der2.shape shouldBe Shape(6, 2, 3, 2, 3)
    }

    @Test fun reverse() {
        val x = FloatTensor(Shape(2, 3)) { it.toFloat() }
        val der0 = f(x)
        val der1 = reverseDerivative1(x, ::f)
        val der2 = reverseDerivative2(x, ::f)
        der0.shape shouldBe Shape(6)
        der1.shape shouldBe Shape(2, 3, 6)
        der2.shape shouldBe Shape(2, 3, 2, 3, 6)
    }

    @Test fun mixed() {
        val x = FloatTensor(Shape(2, 3)) { it.toFloat() }
        val forwardReverseDer = reverseDerivative(x) { y: DTensor -> forwardDerivative(y, ::f) }
        val reverseForwardDer = forwardDerivative(x) { y: DTensor -> reverseDerivative(y, ::f) }
        forwardReverseDer.shape shouldBe Shape(2, 3, 6, 2, 3)
        reverseForwardDer.shape shouldBe Shape(2, 3, 6, 2, 3)
    }

    @Test fun specifyDims() {
        val x = FloatTensor(Shape(2, 3, 4, 5)) { it.toFloat() }
        x.flatten().shape shouldBe Shape (120)
        x.flatten(1).shape shouldBe Shape (2, 60)
        x.flatten(endDim=2).shape shouldBe Shape (24, 5)
        x.flatten(1, 2).shape shouldBe Shape (2, 12, 5)
        x.flatten(1, 1).shape shouldBe Shape (2, 3, 4, 5)
        forwardDerivative1(x.flatten(1, 2), ::f).shape shouldBe Shape (120, 2, 12, 5)
        reverseDerivative1(x.flatten(1, 2), ::f).shape shouldBe Shape (2, 12, 5, 120)
    }
}
