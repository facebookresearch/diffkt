/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import org.diffkt.external.Predicate
import testutils.*

// Also see [RelOpsTest]
class PredicateTest: AnnotationSpec() {

    @Test
    fun testBinaryPredicate() {
        val t1 = tensorOf(1f, 4.3f, 3f, 4.9f, 3f, 2f)
        val t2 = tensorOf(1.3f, 2.5f, 2.8f, 4.9f, 5f, 6f)
        abs(t1 - t2) lt 1f shouldBeExactly tensorOf(1f, 0f, 1f, 1f, 0f, 0f)
    }

    @Test
    fun testIfThenElse() {
        val t1 = tensorOf(1f, 4.3f, 3f, 4.9f, 3f, 2f)
        val t2 = tensorOf(1.3f, 2.5f, 2.8f, 4.9f, 5f, 6f)
        ifThenElse(t1.gt(t2), 2f*t1, t2) shouldBeExactly tensorOf(1.3f, 8.6f, 6f, 4.9f, 5f, 6f)
    }

    @Test
    fun testIfThenElseForwardDerivatives() {
        val t1 = tensorOf(1f, 4.3f, 3f, 4.9f, 3f, 2f)
        val t2 = tensorOf(1.3f, 2.5f, 2.8f, 4.9f, 5f, 6f)
        val (d1, d2) = forwardDerivative(t1, t2) { x, y -> ifThenElse(x.gt(y), 2f*x, y) }
        d1 shouldBeExactly tensorOf(
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 2f, 0f, 0f, 0f, 0f,
                0f, 0f, 2f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f
        ).reshape(Shape(6, 6))
        d2 shouldBeExactly tensorOf(
                1f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 1f, 0f, 0f,
                0f, 0f, 0f, 0f, 1f, 0f,
                0f, 0f, 0f, 0f, 0f, 1f
        ).reshape(Shape(6, 6))
    }

    @Test
    fun testIfThenElseForwardDerivativeSingle() {
        val t1 = tensorOf(1f, 4.3f, 3f, 4.9f, 3f, 2f)
        val t2 = tensorOf(1.3f, 2.5f, 2.8f, 4.9f, 5f, 6f)
        val d1 = forwardDerivative(t1) { x -> ifThenElse(x.gt(t2), 2f*x, t2) }
        val d2 = forwardDerivative(t2) { x -> ifThenElse(t1.gt(x), 2f*t1, x) }
        d1 shouldBeExactly tensorOf(
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 2f, 0f, 0f, 0f, 0f,
                0f, 0f, 2f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f
        ).reshape(Shape(6, 6))
        d2 shouldBeExactly tensorOf(
                1f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 1f, 0f, 0f,
                0f, 0f, 0f, 0f, 1f, 0f,
                0f, 0f, 0f, 0f, 0f, 1f
        ).reshape(Shape(6, 6))
    }

    @Test
    fun testIfThenElseReverseDerivatives() {
        val t1 = tensorOf(1f, 4.3f, 3f, 4.9f, 3f, 2f)
        val t2 = tensorOf(1.3f, 2.5f, 2.8f, 4.9f, 5f, 6f)
        val (d1, d2) = reverseDerivative(t1, t2) { x, y -> ifThenElse(x.gt(y), 2f*x, y) }
        d1 shouldBeExactly tensorOf(
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 2f, 0f, 0f, 0f, 0f,
                0f, 0f, 2f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f
        ).reshape(Shape(6, 6))
        d2 shouldBeExactly tensorOf(
                1f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 1f, 0f, 0f,
                0f, 0f, 0f, 0f, 1f, 0f,
                0f, 0f, 0f, 0f, 0f, 1f
        ).reshape(Shape(6, 6))
    }

    @Test
    fun testIfThenElseReverseDerivativeSingle() {
        val t1 = tensorOf(1f, 4.3f, 3f, 4.9f, 3f, 2f)
        val t2 = tensorOf(1.3f, 2.5f, 2.8f, 4.9f, 5f, 6f)
        val d1 = reverseDerivative(t1) { x -> ifThenElse(x.gt(t2), 2f*x, t2) }
        val d2 = reverseDerivative(t2) { x -> ifThenElse(t1.gt(x), 2f*t1, x) }
        d1 shouldBeExactly tensorOf(
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 2f, 0f, 0f, 0f, 0f,
                0f, 0f, 2f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f
        ).reshape(Shape(6, 6))
        d2 shouldBeExactly tensorOf(
                1f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 0f, 0f, 0f,
                0f, 0f, 0f, 1f, 0f, 0f,
                0f, 0f, 0f, 0f, 1f, 0f,
                0f, 0f, 0f, 0f, 0f, 1f
        ).reshape(Shape(6, 6))
    }

    @Test
    fun testIfGreaterThanZero() {
        val tested = tensorOf(0f, Float.MIN_VALUE, -Float.MIN_VALUE, -0f, Float.MAX_VALUE, Float.MIN_VALUE, 0f, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN)
        val greater = tensorOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)
        val less = -greater
        val expected = tensorOf(-1f, 2f, -3f, -4f, 5f, 6f, -7f, -8f, 9f, -10f)
        ifThenElse(tested, greater, less) shouldBeExactly expected
    }

    @Test
    fun `test that the native implementation of ifThenElse does what we require`() {
        val tested = tensorOf(0f, Float.MIN_VALUE, -Float.MIN_VALUE, -0f, Float.MAX_VALUE, Float.MIN_VALUE, 0f, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN) as StridedFloatTensor
        val greater = tensorOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f) as StridedFloatTensor
        val less = -greater as StridedFloatTensor
        val expected = tensorOf(-1f, 2f, -3f, -4f, 5f, 6f, -7f, -8f, 9f, -10f)
        FloatTensor(greater.shape, Predicate.ifThenElse(
            tested.data,
            greater.data,
            less.data,
            tested.size,
        )) shouldBeExactly expected
    }
}