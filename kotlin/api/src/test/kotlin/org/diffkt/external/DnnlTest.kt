/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.external

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import testutils.floats
import testutils.shouldBeExactly


class DnnlTest : AnnotationSpec() {
    @Test
    fun `check that addition works with strides and offsets`() {
        val t1 = StridedFloatTensor(Shape(5, 40), 2, intArrayOf(40, 1), floats(200+2), StridedUtils.Layout.CUSTOM)
        val t2 = StridedFloatTensor(Shape(5, 40), 3, intArrayOf(1, 5), floats(200+3), StridedUtils.Layout.CUSTOM)
        Dnnl.add(t1, t2) shouldBeExactly (t1.normalize() + (t2.normalize()))
        t1 + t2 shouldBeExactly (t1.normalize() + t2.normalize())
    }

    @Test
    fun `check that subtraction works with strides and offsets`() {
        val t1 = StridedFloatTensor(Shape(5, 40), 2, intArrayOf(40, 1), floats(200+2), StridedUtils.Layout.CUSTOM)
        val t2 = StridedFloatTensor(Shape(5, 40), 3, intArrayOf(1, 5), floats(200+3), StridedUtils.Layout.CUSTOM)
        Dnnl.sub(t1, t2) shouldBeExactly (t1.normalize() - (t2.normalize()))
        t1 - t2 shouldBeExactly (t1.normalize() - t2.normalize())
    }

    @Test
    fun `check that matmul works with strides and offsets`() {
        val t1 = StridedFloatTensor(Shape(2,3), offset = 2, strides = intArrayOf(3, 1), floats(6 + 2), StridedUtils.Layout.CUSTOM)
        val t2 = StridedFloatTensor(Shape(3,4), offset = 3, strides = intArrayOf(1, 3), floats(12 + 3), StridedUtils.Layout.CUSTOM)
        Dnnl.matmul(t1, t2, Shape(), Shape(2), Shape(4)) shouldBeExactly (t1.normalize().matmul(t2.normalize()))
        t1.matmul(t2) shouldBeExactly (t1.normalize().matmul(t2.normalize()))
    }

    @Test
    fun `check scalar mult works with strides and offsets`() {
        val s = 101.3f
        val t = StridedFloatTensor(Shape(3,4), offset = 3, strides = intArrayOf(1, 3), floats(12 + 3), StridedUtils.Layout.CUSTOM)
        Dnnl.mulScalar(t, s) shouldBeExactly (t.normalize() * s)
        t * s shouldBeExactly (t.normalize() * s)
    }
}