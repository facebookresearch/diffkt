/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe
import org.diffkt.StridedUtils.Layout

class StridedFloatTensorTest : AnnotationSpec() {
    @Test
    fun layoutFromShapeStridesTest() {
        val shape = Shape(2, 3, 4)
        val contiguousStrides = intArrayOf(12, 4, 1) // contigStrides(shape)
        val singletonStrides = intArrayOf(0, 0, 0) // zeroStrides(shape.size)
        val repeatingStrides = intArrayOf(0, 0, 1)
        val repeatingStrides2 = intArrayOf(0, 4, 1)
        val customStrides = intArrayOf(1, 4, 12) // contigStrides(shape).reversed()
        val customStrides2 = intArrayOf(12, 0, 0)
        val customStrides3 = intArrayOf(12, 4, 0)

        StridedUtils.layoutFromShapeStrides(-1, shape, contiguousStrides) shouldBe Layout.NATURAL
        StridedUtils.layoutFromShapeStrides(-1, shape, singletonStrides) shouldBe Layout.SINGLETON
        StridedUtils.layoutFromShapeStrides(3, shape, repeatingStrides) shouldBe Layout.CUSTOM
        StridedUtils.layoutFromShapeStrides(4, shape, repeatingStrides) shouldBe Layout.REPEATING
        StridedUtils.layoutFromShapeStrides(5, shape, repeatingStrides) shouldBe Layout.CUSTOM
        StridedUtils.layoutFromShapeStrides(11, shape, repeatingStrides2) shouldBe Layout.CUSTOM
        StridedUtils.layoutFromShapeStrides(12, shape, repeatingStrides2) shouldBe Layout.REPEATING
        StridedUtils.layoutFromShapeStrides(13, shape, repeatingStrides2) shouldBe Layout.CUSTOM
        StridedUtils.layoutFromShapeStrides(-1, shape, customStrides) shouldBe Layout.CUSTOM
        StridedUtils.layoutFromShapeStrides(-1, shape, customStrides2) shouldBe Layout.CUSTOM
        StridedUtils.layoutFromShapeStrides(-1, shape, customStrides3) shouldBe Layout.CUSTOM
    }
}