/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.collections.shouldHaveSize
import io.kotest.matchers.shouldBe
import testutils.*

class IndicesTest : AnnotationSpec() {
    @Test
    fun indices() {
        val shape = Shape(2, 3)
        val x = FloatTensor(shape, floats(6))
        val expected = mutableListOf<List<Int>>()
        for (dim0 in 0 until 2) {
            for (dim1 in 0 until 3) {
                expected.add(listOf(dim0, dim1))
            }
        }
        val xindices = x.indices
        for (i in 0 until 6) {
            xindices.hasNext() shouldBe true
            xindices.next() shouldBe expected[i]
        }
        xindices.hasNext() shouldBe false
    }

    @Test
    fun scalarIndices() {
        val x = FloatScalar(1f)
        val actual = mutableListOf<IntArray>()
        for (ixs in x.indices)
            actual.add(ixs)
        actual shouldHaveSize 1
        actual[0].size shouldBe 0
    }

    // TODO https://github.com/facebookincubator/diffkt/issues/98: test indices with derivatives (forward, reverse)
}

