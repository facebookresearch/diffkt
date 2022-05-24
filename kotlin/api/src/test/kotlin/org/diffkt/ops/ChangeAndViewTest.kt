/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import testutils.*

class ChangeAndViewTest: AnnotationSpec() {
    @Test fun scalarView01() {
        val a = FloatScalar(1F)
        val b = a.view(IntArray(0))
        b shouldBeExactly a
    }
    @Test fun scalarView02() {
        val a = FloatScalar(1F)
        shouldThrow<IllegalArgumentException> {
            a.view(IntArray(1))
        }
    }
    @Test fun sameView01() {
        val a = tensorOf(1F, 2F).reshape(1, 2)
        val b = a.view(IntArray(0))
        b shouldBeExactly a
    }
    @Test fun simpleView01() {
        val a = tensorOf(1F, 2F).reshape(1, 2)
        val b = a.view(intArrayOf(0))
        b shouldBeExactly a.reshape(2)
    }
    @Test fun simpleView02() {
        val a = tensorOf(1F, 2F).reshape(1, 2)
        val b = a.view(intArrayOf(0, 1))
        b shouldBeExactly tensorOf(2F).reshape()
    }
    @Test fun viewOfChange01() {
        val a = tensorOf(1F, 2F, 3F).reshape(1, 3).withChange(intArrayOf(0, 1), FloatScalar(4F))
        a shouldBeExactly tensorOf(1F, 4F, 3F).reshape(1, 3)
        val b = a.view(intArrayOf(0, 0))
        b shouldBeExactly tensorOf(1F).reshape()
        val c = a.view(intArrayOf(0, 1))
        c shouldBeExactly tensorOf(4F).reshape()
        val d = a.view(intArrayOf(0, 2))
        d shouldBeExactly tensorOf(3F).reshape()
    }
    @Test fun viewOfChange02() {
        val a = tensorOf(1F, 2F, 3F).reshape(1, 3).withChange(intArrayOf(0, 1), FloatScalar(4F))
        a shouldBeExactly tensorOf(1F, 4F, 3F).reshape(1, 3)
        val b = a.view(0, 1)
        b shouldBeExactly tensorOf(1F).reshape(1)
        val c = a.view(1, 1)
        c shouldBeExactly tensorOf(4F).reshape(1)
        val e = a.view(2, 1)
        e shouldBeExactly tensorOf(3F).reshape(1)

        val d = a.view(0, 0)
        d shouldBeExactly tensorOf(1F, 4F, 3F).reshape(3)
    }
    @Test fun viewOfChange03() {
        val a = tensorOf(1F, 2F, 3F).reshape(1, 3).withChange(intArrayOf(0, 1), FloatScalar(4F))
        a shouldBeExactly tensorOf(1F, 4F, 3F).reshape(1, 3)
        a.view(0, 0).view(0, 0) shouldBeExactly tensorOf(1F).reshape()
        a.view(0, 0).view(1, 0) shouldBeExactly tensorOf(4F).reshape()
        a.view(0, 0).view(2, 0) shouldBeExactly tensorOf(3F).reshape()
    }
    @Test fun viewOfChange04() {
        val a = tensorOf(1F, 2F, 3F).reshape(1, 3).withChange(intArrayOf(0, 1), FloatScalar(4F))
        a.view(intArrayOf(0)) shouldBeExactly tensorOf(1F, 4F, 3F).reshape(3)
    }
    @Test fun simpleView03() {
        val a = tensorOf(1F, 2F, 3F).reshape(1, 3)
        a.view(0, 0) shouldBeExactly tensorOf(1F, 2F, 3F).reshape(3)
    }
    fun testWithChange(a: DTensor, b: DTensor, index: Int, axis: Int, expected: DTensor) {
        fun m(a: DTensor, b: DTensor) : DTensor {
            val c = a.withChange(index, axis, b)
            c shouldBeExactly expected
            c.view(index, axis) shouldBeExactly b
            c.withChange(index, axis, a.view(index, axis)) shouldBeExactly a
            return c
        }
        m(a, b)
        val aIdentity = identityGradientofSameKind(a)
        val bIdentity = identityGradientofSameKind(b)
        val d1 = aIdentity.withChange(index, axis, zeroOfSameKind(a, aIdentity.shape.remove(axis)))
        val d2 = zeroOfSameKind(a, a.shape + b.shape).withChange(index, axis, bIdentity)
        val d2r = d2.leftTranspose(a.shape, b.shape)
        forwardDerivative(a) { aa -> m(aa, b) } shouldBeExactly d1
        reverseDerivative(a) { aa -> m(aa, b) } shouldBeExactly d1
        forwardDerivative(b) { bb -> m(a, bb) } shouldBeExactly d2
        reverseDerivative(b) { bb -> m(a, bb) } shouldBeExactly d2r
        forwardDerivative(a, b) { aa, bb -> m(aa, bb) } shouldBeExactly Pair(d1, d2)
        reverseDerivative(a, b) { aa, bb -> m(aa, bb) } shouldBeExactly Pair(d1, d2r)
    }
    @Test fun withChange00() {
        val a = tensorOf(
                1F, 2F, 3F, 4F,
                5F, 6F, 7F, 8F,
                9F, 10F, 11F, 12F).reshape(3, 4)
        val b = tensorOf(21F, 22F, 23F)
        val index = 1
        val axis = 1
        val expected = tensorOf(
                1F, 21F, 3F, 4F,
                5F, 22F, 7F, 8F,
                9F, 23F, 11F, 12F).reshape(3, 4)
        testWithChange(a, b, index, axis, expected)
    }
    fun testWithChange(a: DTensor, b: DTensor, index: IntRange, axis: Int, expected: DTensor) {
        fun m(a: DTensor, b: DTensor) : DTensor {
            val c = a.withChange(index, axis, b)
            c shouldBeExactly expected
            c.view(index, axis) shouldBeExactly b
            c.withChange(index, axis, a.view(index, axis)) shouldBeExactly a
            return c
        }
        m(a, b)
        val aIdentity = identityGradientofSameKind(a)
        val bIdentity = identityGradientofSameKind(b)
        val d1 = aIdentity.withChange(index, axis, zeroOfSameKind(a, aIdentity.shape.updated(axis, index.endInclusive - index.start + 1)))
        val d2 = zeroOfSameKind(a, a.shape + b.shape).withChange(index, axis, bIdentity)
        val d2r = d2.leftTranspose(a.shape, b.shape)
        forwardDerivative(a) { aa -> m(aa, b) } shouldBeExactly d1
        reverseDerivative(a) { aa -> m(aa, b) } shouldBeExactly d1
        forwardDerivative(b) { bb -> m(a, bb) } shouldBeExactly d2
        reverseDerivative(b) { bb -> m(a, bb) } shouldBeExactly d2r
        forwardDerivative(a, b) { aa, bb -> m(aa, bb) } shouldBeExactly Pair(d1, d2)
        reverseDerivative(a, b) { aa, bb -> m(aa, bb) } shouldBeExactly Pair(d1, d2r)
    }
    @Test fun withChange01() {
        val a = tensorOf(
                1F, 2F, 3F, 4F,
                5F, 6F, 7F, 8F,
                9F, 10F, 11F, 12F).reshape(3, 4)
        val b = tensorOf(21F, 22F, 23F, 24F, 25F, 26F).reshape(3, 2)
        val index = 1..2
        val axis = 1
        val expected = tensorOf(
                1F, 21F, 22F, 4F,
                5F, 23F, 24F, 8F,
                9F, 25F, 26F, 12F).reshape(3, 4)
        testWithChange(a, b, index, axis, expected)
    }
    @Test fun withChange02() {
        val a = tensorOf(1F, 2F, 3F).withChange(1, 0, FloatScalar(4F))
        a shouldBeExactly tensorOf(1F, 4F, 3F)
    }
    @Test fun withChange03() {
        val a = tensorOf(
                1F, 2F, 3F, 4F,
                5F, 6F, 7F, 8F,
                9F, 10F, 11F, 12F).reshape(3, 4)
        val b = a
        val index = 0..3
        val axis = 1
        val expected = a
        testWithChange(a, b, index, axis, expected)
    }
    @Test fun withChange04() {
        val a = tensorOf(1F, 2F, 3F, 4F).reshape(1, 4)
        val b = tensorOf(5F, 6F, 7F, 8F)
        val index = 0
        val axis = 0
        val expected = b.reshape(1, 4)
        testWithChange(a, b, index, axis, expected)
    }
    @Test fun withChange05() {
        val a = tensorOf(1F, 2F, 3F, 4F).reshape(4, 1)
        val b = tensorOf(5F, 6F, 7F, 8F)
        val index = 0
        val axis = 1
        val expected = b.reshape(4, 1)
        testWithChange(a, b, index, axis, expected)
    }
    @Test fun withChange06() {
        val a = tensorOf(
                1F, 2F, 3F, 4F,
                5F, 6F, 7F, 8F,
                9F, 10F, 11F, 12F).reshape(3, 4)
        val b = tensorOf(21F, 22F, 23F, 24F)
        val index = 1
        val axis = 0
        val expected = tensorOf(
                1F, 2F, 3F, 4F,
                21F, 22F, 23F, 24F,
                9F, 10F, 11F, 12F).reshape(3, 4)
        testWithChange(a, b, index, axis, expected)
    }
    @Test fun withChange07() {
        val a = tensorOf(
                1F, 2F, 3F, 4F,
                5F, 6F, 7F, 8F,
                9F, 10F, 11F, 12F).reshape(3, 4)
        val b = tensorOf(21F, 22F, 23F, 24F, 25F, 26F, 27F, 28F).reshape(2, 4)
        val index = 1..2
        val axis = 0
        val expected = tensorOf(
                1F, 2F, 3F, 4F,
                21F, 22F, 23F, 24F,
                25F, 26F, 27F, 28F).reshape(3, 4)
        testWithChange(a, b, index, axis, expected)
    }
    @Test fun withChange08() {
        val a = tensorOf(1F, 2F, 3F).withChange(1, 0, FloatScalar(4F))
        a shouldBeExactly tensorOf(1F, 4F, 3F)
    }
    @Test fun withChange09() {
        val a = tensorOf(
            1F, 2F, 3F, 4F,
            5F, 6F, 7F, 8F,
            9F, 10F, 11F, 12F).reshape(3, 4)
        val b = tensorOf(21F, 22F)
        val index = 1..2
        val axis = 1
        val expected = tensorOf(
            1F, 21F, 22F, 4F,
            5F, 21F, 22F, 8F,
            9F, 21F, 22F, 12F).reshape(3, 4)

        val c = a.withChange(index, axis, b)
        c shouldBeExactly expected
    }
    fun testWithChange(a: DTensor, b: DTensor, index: IntArray, expected: DTensor) {
        fun m(a: DTensor, b: DTensor) : DTensor {
            val c = a.withChange(index, b)
            c shouldBeExactly expected
            c.view(index) shouldBeExactly b
            c.withChange(index, a.view(index)) shouldBeExactly a
            return c
        }
        m(a, b)
        val aIdentity = identityGradientofSameKind(a)
        val bIdentity = identityGradientofSameKind(b)
        val d1 = aIdentity.withChange(index, zeroOfSameKind(a, aIdentity.shape.drop(b.rank + a.rank)))
        val d2 = zeroOfSameKind(a, a.shape + b.shape).withChange(index, bIdentity)
        val d2r = d2.leftTranspose(a.shape, b.shape)
        forwardDerivative(a) { aa -> m(aa, b) } shouldBeExactly d1
        reverseDerivative(a) { aa -> m(aa, b) } shouldBeExactly d1
        forwardDerivative(b) { bb -> m(a, bb) } shouldBeExactly d2
        reverseDerivative(b) { bb -> m(a, bb) } shouldBeExactly d2r
        forwardDerivative(a, b) { aa, bb -> m(aa, bb) } shouldBeExactly Pair(d1, d2)
        reverseDerivative(a, b) { aa, bb -> m(aa, bb) } shouldBeExactly Pair(d1, d2r)
    }
    @Test fun withChangeDerivative01() {
        val a = tensorOf(1F, 2F, 3F).reshape(1, 3)
        val b: DTensor = FloatScalar(4F)
        val index = intArrayOf(0, 1)
        val expected = tensorOf(1F, 4F, 3F).reshape(1, 3)
        testWithChange(a, b, index, expected)
    }
    @Test fun withChangeDerivative02() {
        val a = tensorOf(1F, 2F, 3F).reshape(1, 3)
        val b: DTensor = FloatScalar(4F).reshape(1)
        val index = 1
        val axis = 1
        val expected = tensorOf(1F, 4F, 3F).reshape(1, 3)
        testWithChange(a, b, index, axis, expected)
    }
    fun testView(a: DTensor, index: Int, axis: Int, expected: DTensor) {
        fun m(a: DTensor): DTensor {
            val t = a.view(index, axis)
            t shouldBeExactly expected
            return t
        }
        m(a)
        val d1 = identityGradientofSameKind(a).view(index, axis)
        val d1r = d1.leftTranspose(a.shape.remove(axis), a.shape)
        forwardDerivative(a) { aa -> m(aa) } shouldBeExactly d1
        reverseDerivative(a) { aa -> m(aa) } shouldBeExactly d1r
    }
    @Test fun view01() {
        val a = tensorOf(1F, 2F, 3F).reshape(1, 3)
        val index = 1
        val axis = 1
        val expected = FloatScalar(2F).reshape(1)
        testView(a, index, axis, expected)
    }
    @Test fun view02() {
        val a = tensorOf(1F, 2F, 3F)
        val index = 1
        val axis = 0
        val expected = FloatScalar(2F)
        testView(a, index, axis, expected)
    }
    fun testView(a: DTensor, index: IntArray, expected: DTensor) {
        fun m(a: DTensor): DTensor {
            val t = a.view(index)
            t shouldBeExactly expected
            return t
        }
        m(a)
        val d1 = identityGradientofSameKind(a).view(index)
        val d1r = d1.leftTranspose(a.shape.drop(index.size), a.shape)
        forwardDerivative(a) { aa -> m(aa) } shouldBeExactly d1
        reverseDerivative(a) { aa -> m(aa) } shouldBeExactly d1r
    }
    @Test fun view03() {
        val a = tensorOf(1F, 2F, 3F).reshape(1, 3)
        val index = intArrayOf(0, 1)
        val expected = FloatScalar(2F)
        testView(a, index, expected)
    }
    @Test fun view04() {
        val a = tensorOf(1F, 2F, 3F).reshape(3, 1)
        val index = intArrayOf(1)
        val expected = FloatScalar(2F).reshape(1)
        testView(a, index, expected)
    }
    @Test fun withChangeRange() {
        val a = tensorOf(
                1F, 2F, 3F, 4F,
                5F, 6F, 7F, 8F,
                9F, 10F, 11F, 12F).reshape(3, 4)
        val b = tensorOf(
                21F, 22F,
                23F, 24F,
                25F, 26F).reshape(3, 2)
        val index = 1..2
        val axis = 1
        val expected = tensorOf(
                1F, 21F, 22F, 4F,
                5F, 23F, 24F, 8F,
                9F, 25F, 26F, 12F).reshape(3, 4)
        testWithChange(a, b, index, axis, expected)
    }
}
