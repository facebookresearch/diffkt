/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe

private val IntTensor.scalar: Int
    get() { return at(0) }

class IntTensorTest : AnnotationSpec() {
    @Test fun create() {
        val shape = Shape(2, 3)
        val a = IntTensor(shape, intArrayOf(1, 2, 3, 1, 2, 3))
        a.shape shouldBe shape
        a.data shouldBe intArrayOf(1, 2, 3, 1, 2, 3)
        a.layout shouldBe StridedUtils.Layout.NATURAL
        a.offset shouldBe 0
        a.strides shouldBe StridedUtils.contigStrides(shape)
    }

    @Test fun index() {
        val a = IntTensor(Shape(2, 3), intArrayOf(1, 2, 3, 10, 20, 30))
        val a0 = a[0]
        a0.shape shouldBe Shape(3)
        a0[0].scalar shouldBe 1
        a0[1].scalar shouldBe 2
        a0[2].scalar shouldBe 3
        a[1][0].scalar shouldBe 10
        a[1][1].scalar shouldBe 20
        a[1][2].scalar shouldBe 30
    }
}
