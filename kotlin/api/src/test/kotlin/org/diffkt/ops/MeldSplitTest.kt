/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe
import io.kotest.matchers.types.shouldBeSameInstanceAs
import org.diffkt.*
import org.diffkt.forward.ForwardDerivativeID
import org.diffkt.reverse.ReverseDerivativeID
import testutils.*

class MeldSplitTest : AnnotationSpec() {
    val x1 = FloatScalar.ONE
    val x2 = tensorOf(2F, 3F, 4F).reshape(1, 3)
    val x3 = tensorOf(5F, 6F, 7F, 8F).reshape(2, 2)
    val x4 = FloatScalar(6F)
    val inputs = listOf(x1, x2, x3, x4)

    fun checkMeldedPrimal(melded: DTensor) {
        val split = melded.split(inputs.map { it.shape })
        for (i in inputs.indices) {
            split[i] shouldBeExactly inputs[i]
            split[i].derivativeID shouldBeSameInstanceAs melded.derivativeID
        }
    }
    fun checkMeldedDerivative(melded: DTensor) {
        val k = inputs.map { it.shape.product() }.sum()
        melded shouldBeExactly FloatTensor(Shape(k, k), { if (it % (k + 1) == 0) 1F else 0F })
    }

    @Test
    fun MeldSplitTest0() {
        val melded = meld(inputs)
        melded.derivativeID shouldBeSameInstanceAs NoDerivativeID
        checkMeldedPrimal(melded)
    }

    @Test
    fun MeldSplitTest1() {
        val melded0 = meld(inputs)
        val pad = primalAndForwardDerivative(melded0) { melded1: DTensor ->
            checkMeldedPrimal(melded1)
            (melded1.derivativeID is ForwardDerivativeID) shouldBe true
            melded1
        }
        pad.first.derivativeID shouldBeSameInstanceAs NoDerivativeID
        pad.second.derivativeID shouldBeSameInstanceAs NoDerivativeID
        checkMeldedPrimal(pad.first)
        checkMeldedDerivative(pad.second)
    }

    @Test
    fun MeldSplitTest2() {
        val melded0 = meld(inputs)
        val pad = primalAndReverseDerivativeTransposed(melded0) { melded1: DTensor ->
            checkMeldedPrimal(melded1)
            (melded1.derivativeID is ReverseDerivativeID) shouldBe true
            melded1
        }
        pad.first.derivativeID shouldBeSameInstanceAs NoDerivativeID
        pad.second.derivativeID shouldBeSameInstanceAs NoDerivativeID
        checkMeldedPrimal(pad.first)
        checkMeldedDerivative(pad.second)
    }
}
