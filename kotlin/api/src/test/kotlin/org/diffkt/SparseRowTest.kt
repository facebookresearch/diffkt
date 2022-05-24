/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import io.kotest.core.spec.style.AnnotationSpec
import testutils.shouldBeExactly

class SparseRowTest: AnnotationSpec() {
    @Test
    fun testSparseRowPlusSparseRow() {
        val t1 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 1f, 2f, 3f, 3f, 7f, 8f, 9f))
        val t2 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 2f, 2f, 2f, 4f, 3f, 3f, 3f))
        val out = t1 + t2
        assert(out is SparseRowFloatTensor)
        out shouldBeExactly tensorOf(
            0f, 0f, 0f,
            3f, 4f, 5f,
            0f, 0f, 0f,
            7f, 8f, 9f,
            3f, 3f, 3f
        ).reshape(5, 3)
    }

    @Test
    fun testSparseRowPlusDense() {
        val t1 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 1f, 2f, 3f, 3f, 7f, 8f, 9f))
        val t2 = FloatTensor(Shape(5, 3)) { 2f }
        val out = t1 + t2
        assert(out !is SparseRowFloatTensor)
        out shouldBeExactly tensorOf(
            2f, 2f, 2f,
            3f, 4f, 5f,
            2f, 2f, 2f,
            9f, 10f, 11f,
            2f, 2f, 2f
        ).reshape(5, 3)
    }

    @Test
    fun testSparseRowAll() {
        val tEven = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 2f, 2f, 2f, 4f, 6f, 6f, 6f))
        val tOdd = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(2f, 1f, 1f, 1f, 4f, 5f, 5f, 5f))
        val out1 = tEven.all { it.toInt() % 2 == 0 }
        val out2 = tOdd.all { it.toInt() % 2  == 0 }
        assert(out1 && !out2)
    }

    @Test
    fun testSparseRowMap() {
        val t1 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 1f, 2f, 3f, 3f, 7f, 8f, 9f))
        val out1 = t1.map { it * 2f }
        val out2 = t1.map { it + 2f }
        assert(out1 is SparseRowFloatTensor)
        assert(out2 !is SparseRowFloatTensor)
        out1 shouldBeExactly tensorOf(
            0f, 0f, 0f,
            2f, 4f, 6f,
            0f, 0f, 0f,
            14f, 16f, 18f,
            0f, 0f, 0f
        ).reshape(5, 3)
        out2 shouldBeExactly tensorOf(
            2f, 2f, 2f,
            3f, 4f, 5f,
            2f, 2f, 2f,
            9f, 10f, 11f,
            2f, 2f, 2f
        ).reshape(5, 3)
    }

    @Test
    fun testSparseRowZip() {
        val t1 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 1f, 2f, 3f, 3f, 7f, 8f, 9f))
        val t2 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 2f, 2f, 2f, 4f, 3f, 3f, 3f))
        val leftAnnihilator = {l: Float, r: Float -> if (l == 0f) 0f else l + r}
        val rightAnnihilator = {l: Float, r: Float -> if (r == 0f) 0f else l + r}
        val out1 = t1.qualifiedZip(t2, true, false, leftAnnihilator)
        val out2 = t1.qualifiedZip(t2, false, true, rightAnnihilator)
        assert(out1 is SparseRowFloatTensor)
        assert(out2 is SparseRowFloatTensor)
        out1 shouldBeExactly tensorOf(
            0f, 0f, 0f,
            3f, 4f, 5f,
            0f, 0f, 0f,
            7f, 8f, 9f,
            0f, 0f, 0f
        ).reshape(5, 3)
        assert((out1 as SparseRowFloatTensor).data.size == 8)
        out2 shouldBeExactly tensorOf(
            0f, 0f, 0f,
            3f, 4f, 5f,
            0f, 0f, 0f,
            0f, 0f, 0f,
            3f, 3f, 3f
        ).reshape(5, 3)
        assert((out2 as SparseRowFloatTensor).data.size == 8)
    }

    @Test
    fun testSparseRowZipZeroTrim() {
        val t1 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 1f, 2f, 3f, 3f, 7f, 8f, 9f))
        val t2 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 2f, 2f, 2f, 4f, 3f, 3f, 3f))
        val f = {l : Float, r: Float -> if (l < 5.0f) 0f else l + r}
        val out = t1.zip(t2, f)
        assert(out is SparseRowFloatTensor)
        out shouldBeExactly tensorOf(
            0f, 0f, 0f,
            0f, 0f, 0f,
            0f, 0f, 0f,
            7f, 8f, 9f,
            0f, 0f, 0f
        ).reshape(5, 3)
        assert((out as SparseRowFloatTensor).data.contentEquals(floatArrayOf(3f, 7f, 8f, 9f)))
    }

    @Test
    fun testSparseRowZipToStrided(){
        val t1 = SparseRowFloatTensor(Shape(4, 3), floatArrayOf(0f, 1f, 2f, 3f, 3f, 7f, 8f, 9f))
        val t2 = SparseRowFloatTensor(Shape(4, 3), floatArrayOf(1f, 2f, 2f, 2f, 4f, 3f, 3f, 3f))
        val out = t1.zip(t2) {l, r -> l + r}
        assert(out is StridedFloatTensor)
        out shouldBeExactly tensorOf(
            1f, 2f, 3f,
            2f, 2f, 2f,
            7f, 8f, 9f,
            3f, 3f, 3f
        ).reshape(4, 3)
        assert((out as StridedFloatTensor).data.contentEquals(floatArrayOf(0f, 1f, 2f, 3f, 1f, 2f, 2f ,2f, 3f, 7f, 8f, 9f, 4f, 3f, 3f, 3f)))
    }

    @Test
    fun testSparseRowTimes() {
        val t1 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 1f, 2f, 3f, 3f, 7f, 8f, 9f))
        val t2 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 2f, 2f, 2f, 4f, 3f, 3f, 3f))
        val out = t1 * t2
        assert(out is SparseRowFloatTensor)
        out shouldBeExactly tensorOf(
            0f, 0f, 0f,
            2f, 4f, 6f,
            0f, 0f, 0f,
            0f, 0f, 0f,
            0f, 0f, 0f
        ).reshape(5, 3)
    }

    @Test
    fun testSparseRowMinus() {
        val t1 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 1f, 2f, 3f, 3f, 7f, 8f, 9f))
        val t2 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 2f, 2f, 2f, 4f, 3f, 3f, 3f))
        val out = t1 - t2
        assert(out is SparseRowFloatTensor)
        out shouldBeExactly tensorOf(
            0f, 0f, 0f,
            -1f, 0f, 1f,
            0f, 0f, 0f,
            7f, 8f, 9f,
            -3f, -3f, -3f
        ).reshape(5, 3)
    }

    @Test
    fun testSparseRowDiv() {
        val t1 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 1f, 2f, 3f, 3f, 7f, 8f, 9f))
        val t2 = SparseRowFloatTensor(Shape(5, 3), floatArrayOf(1f, 2f, 2f, 2f, 4f, 3f, 3f, 3f))
        val out = t1 / t2
        assert(out !is SparseRowFloatTensor)
        out shouldBeExactly tensorOf(
            Float.NaN, Float.NaN, Float.NaN,
            0.5f, 1.0f, 1.5f,
            Float.NaN, Float.NaN, Float.NaN,
            Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY,
            0f, 0f, 0f
        ).reshape(5, 3)
    }
}