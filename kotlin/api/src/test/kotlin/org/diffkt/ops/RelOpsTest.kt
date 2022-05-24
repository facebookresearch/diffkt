/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import testutils.*

class RelOpsTest: AnnotationSpec() {
    @Test
    fun compare() {
        val a = FloatScalar(3f)
        assert(a > 0f)
        assert(a <= 4f)
        val f = { x: DScalar -> x - 2f }
        assert(f(a) < 2f)
        assert(a == a)
        assert(a != FloatScalar(2f))
    }

    @Test
    fun compareTensors() {
        val t1 = tensorOf(1f, 2f, 3f, 4f, 5f, 6f)
        val t2 = tensorOf(3f, 2f, 4f, 5f, 6f, 5f)

        t1.gt(t2) shouldBeExactly tensorOf(0f, 0f, 0f, 0f, 0f, 1f)
        t1.ge(t2) shouldBeExactly tensorOf(0f, 1f, 0f, 0f, 0f, 1f)
        t1.lt(t2) shouldBeExactly tensorOf(1f, 0f, 1f, 1f, 1f, 0f)
        t1.le(t2) shouldBeExactly tensorOf(1f, 1f, 1f, 1f, 1f, 0f)
    }

    @Test
    fun compareTensorScalar() {
        val t = tensorOf(1f, 2f, 3f, 4f, 5f, 6f)
        val x = 3f
        val xScalar = FloatScalar(3f)

        val gtExpected = tensorOf(0f, 0f, 0f, 1f, 1f, 1f)
        val geExpected = tensorOf(0f, 0f, 1f, 1f, 1f, 1f)
        val ltExpected = tensorOf(1f, 1f, 0f, 0f, 0f, 0f)
        val leExpected = tensorOf(1f, 1f, 1f, 0f, 0f, 0f)

        t.gt(x) shouldBeExactly gtExpected
        t.gt(xScalar) shouldBeExactly gtExpected

        t.ge(x) shouldBeExactly geExpected
        t.ge(xScalar) shouldBeExactly geExpected

        t.lt(x) shouldBeExactly ltExpected
        t.lt(xScalar) shouldBeExactly ltExpected

        t.le(x) shouldBeExactly leExpected
        t.le(xScalar) shouldBeExactly leExpected
    }
}