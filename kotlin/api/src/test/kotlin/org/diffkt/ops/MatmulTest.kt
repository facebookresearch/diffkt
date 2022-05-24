/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.FloatTensor
import org.diffkt.Shape
import org.diffkt.matmul
import kotlin.random.Random
import testutils.*

class MatmulTest : AnnotationSpec() {
    @Test fun `vector x vector`() {
        val r = Random(0)
        val tensor1 = FloatTensor.random(r, Shape(3))
        val tensor2 = FloatTensor.random(r, Shape(3))
        val result = tensor1.matmul(tensor2)
        result.shape shouldBe Shape() // dot product, producing a scalar
    }
    @Test fun `matrix x vector`() {
        val r = Random(0)
        val tensor1 = FloatTensor.random(r, Shape(3, 4))
        val tensor2 = FloatTensor.random(r, Shape(4))
        val result = tensor1.matmul(tensor2)
        result.shape shouldBe Shape(3)
    }
    @Test fun `batched matrix x broadcasted vector`() {
        val r = Random(0)
        val tensor1 = FloatTensor.random(r, Shape(10, 3, 4))
        val tensor2 = FloatTensor.random(r, Shape(4))
        val result = tensor1.matmul(tensor2)
        result.shape shouldBe Shape(10, 3)
    }
    @Test fun `batched matrix x batched matrix`() {
        val r = Random(0)
        val tensor1 = FloatTensor.random(r, Shape(10, 3, 4))
        val tensor2 = FloatTensor.random(r, Shape(10, 4, 5))
        val result = tensor1.matmul(tensor2)
        result.shape shouldBe Shape(10, 3, 5)
    }
    @Test fun `batched matrix x broadcasted matrix`() {
        val r = Random(0)
        val tensor1 = FloatTensor.random(r, Shape(10, 3, 4))
        val tensor2 = FloatTensor.random(r, Shape(4, 5))
        val result = tensor1.matmul(tensor2)
        result.shape shouldBe Shape(10, 3, 5)
    }
    @Test fun `vector x matrix`() {
        val r = Random(0)
        val tensor1 = FloatTensor.random(r, Shape(4))
        val tensor2 = FloatTensor.random(r, Shape(4, 5))
        val result = tensor1.matmul(tensor2)
        result.shape shouldBe Shape(5)
    }
    @Test fun `vector x batched matrix`() {
        val r = Random(0)
        val tensor1 = FloatTensor.random(r, Shape(4))
        val tensor2 = FloatTensor.random(r, Shape(10, 4, 5))
        val result = tensor1.matmul(tensor2)
        result.shape shouldBe Shape(10, 5)
    }
    @Suppress("INVALID_STYPE_ERROR") // to test runtime mismatch check
    @Test fun `test shape mismatch`() {
        val r = Random(0)
        val tensor1 = FloatTensor.random(r, Shape(3, 4))
        val tensor2 = FloatTensor.random(r, Shape(5, 6))
        shouldThrow<IllegalArgumentException> { tensor1.matmul(tensor2) }
    }
    @Suppress("INVALID_STYPE_ERROR") // to test runtime mismatch check
    @Test fun `test batch shape mismatch`() {
        val r = Random(0)
        val tensor1 = FloatTensor.random(r, Shape(10, 3, 4))
        val tensor2 = FloatTensor.random(r, Shape(11, 4, 5))
        shouldThrow<IllegalArgumentException> { tensor1.matmul(tensor2) }
    }
}
