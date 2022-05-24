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

class ScatterTest: AnnotationSpec() {
    @Test fun scatter2D() {
        val x = FloatTensor(Shape(2, 5), floats(10))
        val r1 = x.scatter(listOf(1, 2), axis = 0, Shape(3, 5))
        r1 shouldBe FloatTensor(
            Shape(3, 5),
            0f, 0f, 0f, 0f, 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)
        assert(r1 is SparseRowFloatTensor)
        val r2 = x.scatter(listOf(0, 1, 2), axis = 1, Shape(3, 5))
        r2 shouldBe FloatTensor(
            Shape(3, 5),
            1f, 2f, 3f, 0f, 0f, 6f, 7f, 8f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        assert(r2 !is SparseRowFloatTensor)
        val r3 = x.scatter(listOf(1, 0), axis = 0, Shape(2, 5))
        assert(r3 !is SparseRowFloatTensor)
        r3 shouldBe FloatTensor(Shape(2, 5), 6f, 7f, 8f, 9f, 10f, 1f, 2f, 3f, 4f, 5f)
    }

    @Test fun scatter3D() {
        val x = FloatTensor(Shape(2, 2, 2), floats(8))
        x.scatter(listOf(0, 2), axis = 0, Shape(3, 2, 2)) shouldBe FloatTensor(
            Shape(3, 2, 2),
             1f, 2f, 3f, 4f, 0f, 0f, 0f, 0f, 5f, 6f, 7f, 8f)

        x.scatter(listOf(1), axis = 1, Shape(3, 2, 2)) shouldBe FloatTensor(
            Shape(3, 2, 2),
             0f, 0f, 1f, 2f, 0f, 0f, 5f, 6f, 0f, 0f, 0f, 0f)

        x.scatter(listOf(0, 1), axis = 2, Shape(3, 2, 2)) shouldBe FloatTensor(
            Shape(3, 2, 2),
            1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 0f, 0f, 0f, 0f)
    }

    @Test fun scatterAtIndices() {
        val x = FloatTensor(Shape(2, 3), floats(6))
        x.operations.scatterAtIndices(x, listOf(intArrayOf(0)), Shape(4,6)) shouldBe FloatTensor(
            Shape(4, 6),
             1f, 2f, 3f, 4f, 5f, 6f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
        x.operations.scatterAtIndices(x, listOf(intArrayOf(0, 1), intArrayOf(0, 1)), Shape(2, 4,3)) shouldBe FloatTensor(
            Shape(2, 4, 3),
             0f, 0f, 0f, 4f, 5f, 6f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f)
    }

    @Test fun scatterReverse2D() {
        val x = FloatTensor(Shape(2, 5), floats(10))
        reverseDerivative(x) { xx -> xx.scatter(listOf(1, 2), axis = 0, Shape(3, 5)) } shouldBe
                FloatTensor(
                    Shape(2, 5, 3, 5),
                    zeros(5) + ones(1) + (zeros(15) + ones(1)) * 9)
    }

    @Test fun scatterReverse3D() {
        val x = FloatTensor(Shape(1, 2, 1), floats(2))
        reverseDerivative(x) { xx -> xx.scatter(listOf(0), axis = 0, Shape(2, 2, 1)) } shouldBe
                FloatTensor(
                    Shape(1, 2, 1, 2, 2, 1),
                    1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f)
        reverseDerivative(x) { xx -> xx.scatter(listOf(0), axis = 2, Shape(1, 2, 2)) } shouldBe
                FloatTensor(
                    Shape(1, 2, 1, 1, 2, 2),
                    1f, 0f, 0f, 0f, 0f, 0f, 1f, 0f)
    }

    @Test fun scatterForward2D() {
        val x = FloatTensor(Shape(2, 3), floats(6))
        forwardDerivative(x) { xx -> xx.scatter(listOf(0), axis = 0, Shape(3, 3)) } shouldBe
                FloatTensor(
                    Shape(3, 3, 2, 3),
                    (ones(1) + zeros(6)) * 2 + ones(1) + zeros(39))
    }

    @Test fun scatterForward3D() {
        val x = FloatTensor(Shape(1, 2, 1), floats(2))
        forwardDerivative(x) { xx -> xx.scatter(listOf(0), axis = 0, Shape(2, 2, 1)) } shouldBe
                FloatTensor(
                    Shape(2, 2, 1, 1, 2, 1),
                    1f, 0f, 0f, 1f, 0f, 0f, 0f, 0f)
        forwardDerivative(x) { xx -> xx.scatter(listOf(0), axis = 2, Shape(1, 2, 2)) } shouldBe
                FloatTensor(
                    Shape(1, 2, 2, 1, 2, 1),
                    1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f)
    }
}