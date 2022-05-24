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

class SumTest : AnnotationSpec() {
    @Test fun forwardDefault() {
        val x = FloatTensor(Shape(2, 3)) { it.toFloat() }
        fun f(x: DTensor): DTensor = x.sum()
        val der0 = f(x)
        val der1 = forwardDerivative1(x, ::f)
        val der2 = forwardDerivative2(x, ::f)
        der0 shouldBeExactly FloatTensor(Shape()) { 15f }
        der1 shouldBeExactly FloatTensor(der0.shape + x.shape) { 1f }
        der2 shouldBeExactly zeroOfSameKind(x, der1.shape + x.shape)
    }

    @Test fun forwardExplicitAxes() {
        val x = FloatTensor(Shape(2, 3)) { it.toFloat() }
        fun f(x: DTensor): DTensor = x.sum(intArrayOf(1))
        val der0 = f(x)
        val der1 = forwardDerivative1(x, ::f)
        val der2 = forwardDerivative2(x, ::f)
        der0 shouldBeExactly FloatTensor(Shape(2), floatArrayOf(3f, 12f))
        val threeOnes = floatArrayOf(1f, 1f, 1f)
        val threeZeros = floatArrayOf(0f, 0f, 0f)
        der1 shouldBeExactly FloatTensor(der0.shape + x.shape, threeOnes + threeZeros * 2 + threeOnes)
        der2 shouldBeExactly zeroOfSameKind(x, der1.shape + x.shape)
    }

    @Test fun forwardKeepDims() {
        val x = FloatTensor(Shape(2, 3)) { it.toFloat() }
        fun f(x: DTensor): DTensor = x.sum(keepDims = true)
        val der0 = f(x)
        val der1 = forwardDerivative1(x, ::f)
        val der2 = forwardDerivative2(x, ::f)
        der0 shouldBeExactly FloatTensor(Shape(1, 1)) { 15f }
        der1 shouldBeExactly FloatTensor(der0.shape + x.shape) { 1f }
        der2 shouldBeExactly zeroOfSameKind(x, der1.shape + x.shape)
    }

    @Test fun reverseDefault() {
        val x = FloatTensor(Shape(2, 3)) { it.toFloat() }
        fun f(x: DTensor): DTensor = x.sum()
        val der0 = f(x)
        val der1 = reverseDerivative1(x, ::f)
        val der2 = reverseDerivative2(x, ::f)
        der0 shouldBeExactly FloatTensor(Shape()) { 15f }
        der1 shouldBeExactly FloatTensor(x.shape + der0.shape) { 1f }
        der2 shouldBeExactly zeroOfSameKind(x, x.shape + der1.shape)
    }

    @Test fun reverseExplicitAxes() {
        val x = FloatTensor(Shape(2, 3)) { it.toFloat() }
        fun f(x: DTensor): DTensor = x.sum(intArrayOf(1))
        val der0 = f(x)
        val der1 = reverseDerivative1(x, ::f)
        val der2 = reverseDerivative2(x, ::f)
        der0 shouldBeExactly FloatTensor(Shape(2), floatArrayOf(3f, 12f))
        val oneZero = floatArrayOf(1f, 0f)
        val zeroOne = floatArrayOf(0f, 1f)
        der1 shouldBeExactly FloatTensor(x.shape + der0.shape, oneZero * 3 + zeroOne * 3)
        der2 shouldBeExactly zeroOfSameKind(x, x.shape + der1.shape)
    }

    @Test fun reverseKeepDims() {
        val x = FloatTensor(Shape(2, 3)) { it.toFloat() }
        fun f(x: DTensor): DTensor = x.sum(keepDims = true)
        val der0 = f(x)
        val der1 = reverseDerivative1(x, ::f)
        val der2 = reverseDerivative2(x, ::f)
        der0 shouldBeExactly FloatTensor(Shape(1, 1)) { 15f }
        der1 shouldBeExactly FloatTensor(x.shape + der0.shape) { 1f }
        der2 shouldBeExactly zeroOfSameKind(x, x.shape + der1.shape)
    }
}