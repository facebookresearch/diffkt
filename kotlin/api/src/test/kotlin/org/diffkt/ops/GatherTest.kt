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
import testutils.floats

class GatherTest : AnnotationSpec() {
    @Test fun gather2D() {
        val x = FloatTensor(Shape(2, 3), floats(6))
        x.gather(listOf(1), axis = 0) shouldBe FloatTensor(Shape(1, 3), 4f, 5f, 6f)
        x.gather(listOf(0, 2), axis = 1) shouldBe FloatTensor(Shape(2, 2), 1f, 3f, 4f, 6f)
        x.gather(listOf(0, 2, 1, 0), axis = 1) shouldBe FloatTensor(Shape(2, 4), 1f, 3f, 2f, 1f, 4f, 6f, 5f, 4f)
    }

    @Test fun gather3D() {
        val x = FloatTensor(Shape(3, 2, 3), floats(18))
        x.gather(listOf(1), axis = 0) shouldBe FloatTensor(Shape(1, 2, 3), 7f, 8f, 9f, 10f, 11f, 12f)
        x.gather(listOf(1, 2), axis = 0) shouldBe FloatTensor(
            Shape(2, 2, 3),
            7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f, 16f, 17f, 18f)
        x.gather(listOf(0), axis = 1) shouldBe FloatTensor(
            Shape(3, 1, 3),
            1f, 2f, 3f, 7f, 8f, 9f, 13f, 14f, 15f)
        x.gather(listOf(1), axis = 1) shouldBe FloatTensor(
            Shape(3, 1, 3),
            4f, 5f, 6f, 10f, 11f, 12f, 16f, 17f, 18f)
        x.gather(listOf(0), axis = 2) shouldBe FloatTensor(
            Shape(3, 2, 1),
             1f, 4f, 7f, 10f, 13f, 16f)
        x.gather(listOf(1, 2), axis = 2) shouldBe FloatTensor(
            Shape(3, 2, 2),
             2f, 3f, 5f, 6f, 8f, 9f, 11f, 12f, 14f, 15f, 17f, 18f)
    }

    @Test fun gatherAtIndices() {
        val x = FloatTensor(Shape(2, 3), floats(6))
        x.operations.gatherAtIndices(x, listOf(intArrayOf(0))) shouldBe FloatTensor(Shape(1, 3), 1f, 2f, 3f)
        x.operations.gatherAtIndices(x, listOf(intArrayOf(1))) shouldBe FloatTensor(Shape(1, 3), 4f, 5f, 6f)
        x.operations.gatherAtIndices(x, listOf(intArrayOf(0, 2), intArrayOf(0, 2))) shouldBe FloatTensor(Shape(2), 3f, 3f)
    }

    @Suppress("NAME_SHADOWING")
    @Test fun gatherReverse2D() {
        val x = FloatTensor(Shape(2, 3), floats(6))
        reverseDerivative(x) { x -> x.gather(listOf(1), axis = 0) } shouldBe
                FloatTensor(
                    Shape(2, 3, 1, 3),
                    zeros(9) + ones(1) + (zeros(3) + ones(1)) * 2
                )
        reverseDerivative(x) { x -> x.gather(listOf(0, 2, 1), axis = 1) } shouldBe
                FloatTensor(
                    Shape(2, 3, 2, 3),
                ones(1) + zeros(7) + ones(1) + zeros(4) + ones(1) + zeros(7) + ones(1) + zeros(7) +
                        ones(1) + zeros(4) + ones(1) + zeros(1))
    }

    @Suppress("NAME_SHADOWING")
    @Test fun gatherReverse3D() {
        val x = FloatTensor(Shape(1, 2, 1), floats(2))
        reverseDerivative(x) { x -> x.gather(listOf(0), axis = 0) } shouldBe
                FloatTensor(
                    Shape(1, 2, 1, 1, 2, 1),
                    1f, 0f, 0f, 1f)
        reverseDerivative(x) { x -> x.gather(listOf(0), axis = 1) } shouldBe
                FloatTensor(
                    Shape(1, 2, 1, 1, 1, 1),
                    1f, 0f)
        reverseDerivative(x) { x -> x.gather(listOf(0), axis = 2) } shouldBe
                FloatTensor(
                    Shape(1, 2, 1, 1, 2, 1),
                    1f, 0f, 0f, 1f)

    }

    @Suppress("NAME_SHADOWING")
    @Test fun gatherForward2D() {
        val x = FloatTensor(Shape(2, 3), floats(6))
        forwardDerivative(x) { x -> x.gather(listOf(1), axis = 0) } shouldBe
                FloatTensor(
                    Shape(1, 3, 2, 3),
                    zeros(3) + ones(1) + (zeros(6) + ones(1)) * 2
                )
        forwardDerivative(x) { x -> x.gather(listOf(0, 2, 1), axis = 1) } shouldBe
                FloatTensor(
                    Shape(2, 3, 2, 3),
                    ones(1) + zeros(7) + ones(1) + zeros(4) + ones(1) + zeros(7) + ones(1) + zeros(7) +
                            ones(1) + zeros(4) + ones(1) + zeros(1))
    }

    @Suppress("NAME_SHADOWING")
    @Test fun gatherForward3D() {
        val x = FloatTensor(Shape(1, 2, 1), floats(2))
        forwardDerivative(x) { x -> x.gather(listOf(0), axis = 0) } shouldBe
                FloatTensor(
                    Shape(1, 2, 1, 1, 2, 1),
                    1f, 0f, 0f, 1f)
        forwardDerivative(x) { x -> x.gather(listOf(0), axis = 1) } shouldBe
                FloatTensor(
                    Shape(1, 1, 1, 1, 2, 1),
                    1f, 0f)
        forwardDerivative(x) { x -> x.gather(listOf(0), axis = 2) } shouldBe
                FloatTensor(
                    Shape(1, 2, 1, 1, 2, 1),
                    1f, 0f, 0f, 1f)

    }
}