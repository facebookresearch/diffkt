/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.data.forAll
import io.kotest.data.row
import org.diffkt.*
import testutils.shouldBeExactly

fun zeros(n: Int): FloatArray = FloatArray(n)
fun ones(n: Int): FloatArray = FloatArray(n) { 1f }
operator fun FloatArray.times(n: Int) = FloatArray(n * size) { this[it % size] }

class BroadcastTest : AnnotationSpec() {
    @Test suspend fun forwardAndReverse() {
        forAll(
            row(Shape(), Shape(2), ones(2)),
            row(Shape(1), Shape(2), ones(2)),
            row(Shape(2), Shape(3, 2), floatArrayOf(1f, 0f, 0f, 1f) * 3),
            row(Shape(2, 1), Shape(2, 2), floatArrayOf(1f, 0f, 1f, 0f, 0f, 1f, 0f, 1f)),
        ) { inDims, outDims, der1Data ->
            val x = FloatTensor(inDims) { 5f }
            fun f(x: DTensor) = x.broadcastTo(outDims)

            val der1 = FloatTensor(outDims + inDims, der1Data)
            val der2 = FloatTensor(outDims + inDims + inDims) { 0f }

            f(x) shouldBeExactly FloatTensor(outDims) { 5f }
            forwardDerivative(x, ::f) shouldBeExactly der1
            forwardDerivative2(x, ::f) shouldBeExactly der2

            reverseDerivative(x, ::f) shouldBeExactly der1.leftTranspose(outDims, inDims)
            // Reshape (instead of transpose) is only valid because the values are all the same
            reverseDerivative2(x, ::f) shouldBeExactly der2.reshape(inDims + inDims + outDims)
        }
    }

    /** Basic check that the broadcasting plus primal is as expected. */
    @Test fun broadcastingPlus() {
        val x = FloatTensor(Shape(1, 3)) { it.toFloat() }
        val y = FloatTensor(Shape(2, 2, 3), (zeros(3) + ones(3)) * 2)
        x + y shouldBeExactly(FloatTensor(Shape(2, 2, 3), floatArrayOf(0f, 1f, 2f, 1f, 2f, 3f) * 2))
    }
}
