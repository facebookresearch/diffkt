/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.ints.shouldBeLessThan
import org.diffkt.*
import testutils.shouldBeExactly

class FlipTest: AnnotationSpec() {
    @Test
    fun testFlip01() {
        val shape = Shape(2, 3, 4)
        val input = FloatTensor(shape, { it.toFloat() + 1 })
        input shouldBeExactly FloatTensor(shape,
            1f, 2f, 3f, 4f,
            5f, 6f, 7f, 8f,
            9f, 10f, 11f, 12f,

            13f, 14f, 15f, 16f,
            17f, 18f, 19f, 20f,
            21f, 22f, 23f, 24f,
        )
        input.flip(0) shouldBeExactly FloatTensor(shape,
            13f, 14f, 15f, 16f,
            17f, 18f, 19f, 20f,
            21f, 22f, 23f, 24f,

            1f, 2f, 3f, 4f,
            5f, 6f, 7f, 8f,
            9f, 10f, 11f, 12f,
        )
        input.flip(1) shouldBeExactly FloatTensor(shape,
            9f, 10f, 11f, 12f,
            5f, 6f, 7f, 8f,
            1f, 2f, 3f, 4f,

            21f, 22f, 23f, 24f,
            17f, 18f, 19f, 20f,
            13f, 14f, 15f, 16f,
        )
        input.flip(2) shouldBeExactly FloatTensor(shape,
            4f, 3f, 2f, 1f,
            8f, 7f, 6f, 5f,
            12f, 11f, 10f, 9f,

            16f, 15f, 14f, 13f,
            20f, 19f, 18f, 17f,
            24f, 23f, 22f, 21f,
        )
    }

    @Test
    fun `test indexing when strides are negative`() {
        val shape = Shape(2, 3, 4)
        val input = FloatTensor(shape, { it.toFloat() + 1 })
        val flipped = input.flip(1) as StridedFloatTensor
        flipped.strides[1] shouldBeLessThan 0
        val expected = FloatTensor(shape,
            9f, 10f, 11f, 12f,
            5f, 6f, 7f, 8f,
            1f, 2f, 3f, 4f,

            21f, 22f, 23f, 24f,
            17f, 18f, 19f, 20f,
            13f, 14f, 15f, 16f,
        )
        for (index in flipped.indices) {
            flipped.get(*index) shouldBeExactly expected.get(*index)
            index.fold(flipped as DTensor) { r, i -> r[i] } shouldBeExactly index.fold(expected as DTensor) { r, i -> r[i] }
         }
    }
}
