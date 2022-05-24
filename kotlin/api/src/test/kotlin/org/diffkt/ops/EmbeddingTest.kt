/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe
import org.diffkt.*
import testutils.floats

class EmbeddingTest : AnnotationSpec() {
    @Test
    fun embedding() {
        val table = makeTable(4, 5)
        val input = intTensorOf(0, 2, 3)
        val (res, tableGrad) = primalAndVjp(table, { res -> FloatTensor.ones(res.shape) }) { table_ -> embedding(table_, input) }

        res shouldBe FloatTensor(Shape(3, 5), floats(5) + floats(10, start = 11))
        tableGrad shouldBe FloatTensor(Shape(4, 5), ones(5) + zeros(5) + ones(10))
        assert(tableGrad is SparseRowFloatTensor)
    }

    @Test
    fun threeDimInput() {
        val table = makeTable(10, 5)
        val input = IntTensor(Shape(2, 3), intArrayOf(0, 1, 2, 3, 4, 6))
        val (res, pb) = primalAndPullback(table) { table_ -> embedding(table_, input) }
        val tableGrad = pb(FloatTensor.ones(res.shape))

        res shouldBe FloatTensor(Shape(2, 3, 5), floats(5 * 5) + floats(5, start = 31))
        tableGrad shouldBe FloatTensor(Shape(10, 5), ones(5 * 5) + zeros(5) + ones(5) + zeros(15))
        assert(tableGrad is SparseRowFloatTensor)
    }

    @Test
    fun oneDimInput() {
        val table = makeTable(4, 5)
        val input = IntTensor(Shape(), intArrayOf(1))
        val (res, tableGrad) = primalAndVjp(table, { res -> FloatTensor.ones(res.shape) }) { table_ -> embedding(table_, input) }

        res shouldBe FloatTensor(Shape(5), floats(5, start = 6))
        tableGrad shouldBe FloatTensor(Shape(4, 5), zeros(5) + ones(5) + zeros(10))
        assert(tableGrad !is SparseRowFloatTensor)
    }

    @Test
    fun paddingIndex() {
        val table = makeTable(4, 5)
        val input = IntTensor(Shape(6), intArrayOf(0, 3, 0, 0, 0, 2))
        val (res, tableGrad) = primalAndVjp(table, { res -> FloatTensor.ones(res.shape) }) { table_ -> embedding(table_, input, paddingIndex = 0) }

        res shouldBe FloatTensor(Shape(6, 5), zeros(5) + floats(5, start = 16) + zeros(15) + floats(5, start = 11))
        tableGrad shouldBe FloatTensor(Shape(4, 5), zeros(10) + ones(10))
        assert(tableGrad is SparseRowFloatTensor)
    }

    /** Ensure proper handling of selecting the same embedding vector multiple times */
    @Test
    fun multiSelect() {
        val table = makeTable(4, 5)
        val input = IntTensor(Shape(6), intArrayOf(0, 3, 0, 0, 0, 3))
        val (res, tableGrad) = primalAndVjp(table, { res -> FloatTensor.ones(res.shape) }) { table_ -> embedding(table_, input) }

        val embedding0 = floats(5)
        val embedding3 = floats(5, start = 16)
        val expectedData = embedding0 + embedding3 + embedding0 + embedding0 + embedding0 + embedding3
        res shouldBe FloatTensor(Shape(6, 5), expectedData)
        tableGrad shouldBe FloatTensor(Shape(4, 5), FloatArray(5) { 4f } + zeros(10) + FloatArray(5) { 2f })
        assert(tableGrad is SparseRowFloatTensor)
    }

    @Test
    fun badIndices() {
        val table = makeTable(4, 5)
        val tooLargeIndex = intScalarOf(4)
        val tooSmallIndex = intScalarOf(-1)
        shouldThrow<IllegalArgumentException> { embedding(table, tooLargeIndex) }
        shouldThrow<IllegalArgumentException> { embedding(table, tooSmallIndex) }
    }

    private fun makeTable(
        numEmbeddings: Int,
        embeddingSize: Int
    ): FloatTensor {
        val shape = Shape(numEmbeddings, embeddingSize)
        return FloatTensor(shape, floats(shape.product()))
    }
}
