/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import io.kotest.assertions.collectOrThrow
import io.kotest.assertions.errorCollector
import io.kotest.assertions.failure
import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.booleans.shouldBeFalse
import io.kotest.matchers.booleans.shouldBeTrue
import io.kotest.matchers.ints.shouldBeExactly
import io.kotest.matchers.shouldBe
import io.kotest.matchers.types.shouldBeSameInstanceAs
import io.kotest.matchers.types.shouldNotBeSameInstanceAs
import testutils.*

class ShapeTest : AnnotationSpec() {
    @Test
    fun misShapen() {
        shouldThrow<IllegalArgumentException> {
            tensorOf(1F, 2F, 3F, 4F, 5F, 6F).reshape(1,1)
        }
    }
    @Test
    fun testCtor() {
        val a = intArrayOf(1, 2, 3)
        val s1 = Shape(a.toList())
        val s2 = Shape(1, 2, 3)
        s1 shouldNotBeSameInstanceAs s2
        s1 shouldBe s2
    }
    @Test
    fun testPrepend() {
        val s1 = Shape(2, 3)
        s1.prepend(1) shouldBe Shape(1, 2, 3)
    }
    @Test
    fun testPlus_1() {
        val s1 = Shape(1, 2, 3)
        val s2 = Shape(4, 5, 6)
        s1 + s2 shouldBe Shape(1, 2, 3, 4, 5, 6)
    }
    @Test
    fun testPlus_2() {
        val s1 = Shape()
        val s2 = Shape(4, 5, 6)
        s1 + s2 shouldBe Shape(4, 5, 6)
        s1 + s2 shouldBeSameInstanceAs s2
    }
    @Test
    fun testPlus_3() {
        val s1 = Shape(1, 2, 3)
        val s2 = Shape()
        s1 + s2 shouldBe Shape(1, 2, 3)
        s1 + s2 shouldBeSameInstanceAs s1
    }
    @Test
    fun testPlus_4() {
        val s1 = Shape(1, 2, 3)
        s1 + 4 shouldBe Shape(1, 2, 3, 4)
    }
    @Test
    fun testPlus_5() {
        val s1 = Shape()
        s1 + 4 shouldBe Shape(4)
    }
    @Test
    fun testReversed_0() {
        val s1 = Shape()
        s1.reversed() shouldBe Shape()
    }
    @Test
    fun testReversed_1() {
        val s1 = Shape(1)
        s1.reversed() shouldBe Shape(1)
    }
    @Test
    fun testReversed_2() {
        val s1 = Shape(1, 2, 3)
        s1.reversed() shouldBe Shape(3, 2, 1)
    }
    @Test
    fun testProduct() {
        Shape().product shouldBeExactly 1
        Shape(2).product shouldBeExactly 2
        Shape(1, 2, 3).product shouldBeExactly 6
    }
    @Test
    fun testIsEmpty() {
        Shape().isScalar.shouldBeTrue()
        Shape(2).isScalar.shouldBeFalse()
        Shape(1, 2, 3).isScalar.shouldBeFalse()
    }
    @Test
    fun testTake() {
        Shape().take(0) shouldBe Shape()
        Shape(1, 2).take(0) shouldBe Shape()
        Shape(1, 2).take(1) shouldBe Shape(1)
        Shape(1, 2).take(2) shouldBe Shape(1, 2)
    }
    @Test
    fun testDrop() {
        Shape().drop(0) shouldBe Shape()
        Shape(1, 2).drop(0) shouldBe Shape(1, 2)
        Shape(1, 2).drop(1) shouldBe Shape(2)
        Shape(1, 2).drop(2) shouldBe Shape()
    }
    @Test
    fun testDropLast() {
        Shape().dropLast(0) shouldBe Shape()
        Shape(1, 2).dropLast(0) shouldBe Shape(1, 2)
        Shape(1, 2).dropLast(1) shouldBe Shape(1)
        Shape(1, 2).dropLast(2) shouldBe Shape()
    }
    @Test
    fun testUpdated() {
        Shape(1).updated(0, 3) shouldBe Shape(3)
        Shape(1, 2).updated(0, 3) shouldBe Shape(3, 2)
        Shape(1, 2).updated(1, 3) shouldBe Shape(1, 3)
    }
    @Test
    fun constructWithInvalidDim() {
        val e1 = shouldThrow<IllegalArgumentException> { Shape(0, 1) }
        e1.message shouldBe "Cannot create a shape with dims Shape(0, 1) because it contains a value <= 0"

        val e2 = shouldThrow<IllegalArgumentException> { Shape(-1, 1) }
        e2.message shouldBe "Cannot create a shape with dims Shape(-1, 1) because it contains a value <= 0"
    }
    @Test
    fun takeOutOfBoundRank() {
        shouldThrow<IllegalArgumentException> { Shape(1, 2).take(3) }
        shouldThrow<IllegalArgumentException> { Shape(1).take(-1) }
    }
    @Test
    fun dropOutOfBoundRank() {
        shouldThrow<IllegalArgumentException> { Shape(1, 2).drop(3) }
        shouldThrow<IllegalArgumentException> { Shape(1).drop(-1) }
    }
    @Test
    fun updatedOutOfBoundAxis() {
        val e1 = shouldThrow<IndexOutOfBoundsException> { Shape(1).updated(1, 3) }
        e1.message shouldBe "index 1 out of bounds 0 until 1"

        val e2 = shouldThrow<IndexOutOfBoundsException> { Shape(1).updated(-1, 3) }
        e2.message shouldBe "index -1 out of bounds 0 until 1"
    }
}
