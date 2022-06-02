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

class InnerProductTest : AnnotationSpec() {
    @Test fun innerProduct0() {
        val x = tensorOf(1F, 2F, 3F, 4F, 5F, 6F).reshape(2, 3)
        val y = tensorOf(1F, 2F, 3F, 4F, 5F, 6F, 7F, 8F, 9F, 10F, 11F, 12F, 13F, 14F, 15F).reshape(3, 5)

        fun f(xx: DTensor, yy: DTensor): DTensor {
            val result = xx.innerProduct(Shape(3), yy)
            return result
        }

        val expectedPrimal = tensorOf(46F, 52F, 58F, 64F, 70F, 100F, 115F, 130F, 145F, 160F).reshape(2, 5)
        f(x, y) shouldBeExactly expectedPrimal

        val d1 = tensorOf(
                1.0F, 6.0F, 11.0F, 0.0F, 0.0F, 0.0F, 2.0F, 7.0F, 12.0F, 0.0F,
                0.0F, 0.0F, 3.0F, 8.0F, 13.0F, 0.0F, 0.0F, 0.0F, 4.0F, 9.0F,
                14.0F, 0.0F, 0.0F, 0.0F, 5.0F, 10.0F, 15.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 1.0F, 6.0F, 11.0F, 0.0F, 0.0F, 0.0F, 2.0F,
                7.0F, 12.0F, 0.0F, 0.0F, 0.0F, 3.0F, 8.0F, 13.0F, 0.0F, 0.0F,
                0.0F, 4.0F, 9.0F, 14.0F, 0.0F, 0.0F, 0.0F, 5.0F, 10.0F, 15.0F).reshape(2, 5, 2, 3)
        val d2 = tensorOf(
                1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                3.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 3.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 3.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F, 3.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 3.0F, 4.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                5.0F, 0.0F, 0.0F, 0.0F, 0.0F, 6.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 4.0F, 0.0F, 0.0F, 0.0F, 0.0F, 5.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 6.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 4.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 0.0F, 6.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 4.0F, 0.0F, 0.0F, 0.0F, 0.0F, 5.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 6.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 4.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 0.0F, 6.0F).reshape(2, 5, 3, 5)
        primalAndForwardDerivative(x, y, ::f) shouldBeExactly Pair(expectedPrimal, Pair(d1, d2))
        primalAndReverseDerivativeTransposed(x, y, ::f) shouldBeExactly Pair(expectedPrimal, Pair(d1, d2))

        /*
        primalAndForwardDerivative(x, y) { xx: DTensor, yy: DTensor -> primalAndForwardDerivative(xx, yy, ::f).second }
        primalAndForwardDerivative(x, y) { xx: DTensor, yy: DTensor -> primalAndForwardDerivative(xx, yy, ::f).third }
        primalAndForwardDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndTransposedReverseDerivative(xx, yy, ::f).second })
        primalAndForwardDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndTransposedReverseDerivative(xx, yy, ::f).third })
        primalAndTransposedReverseDerivative(x, y, ::f)
        primalAndTransposedReverseDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndForwardDerivative(xx, yy, ::f).second.first })
        primalAndTransposedReverseDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndForwardDerivative(xx, yy, ::f).second.second })
        primalAndTransposedReverseDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndTransposedReverseDerivative(xx, yy, ::f).second.first })
        primalAndTransposedReverseDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndTransposedReverseDerivative(xx, yy, ::f).second.second })
         */
        // TODO https://github.com/facebookincubator/diffkt/issues/99: should test the values of the derivatives
    }

    @Test fun innerProduct1() {
        val x = tensorOf(1F, 2F, 3F, 4F, 5F, 6F, 7F, 8F, 9F, 10F, 11F, 12F).reshape(2, 3, 2)
        val y = tensorOf(
            1F, 2F, 3F, 4F, 5F, 6F, 7F, 8F, 9F, 10F, 11F, 12F, 13F, 14F, 15F,
            16F, 17F, 18F, 19F, 20F, 21F, 22F, 23F, 24F, 25F, 26F, 27F, 28F, 29F, 30F).reshape(3, 2, 5)

        fun f(xx: DTensor, yy: DTensor): DTensor {
            return xx.innerProduct(Shape(3, 2), yy)
        }

        val expectedPrimal = tensorOf(371F, 392F, 413F, 434F, 455F, 857F, 914F, 971F, 1028F, 1085F).reshape(2, 5)
        f(x, y) shouldBeExactly expectedPrimal

        val d1 = tensorOf(
                1.0F, 6.0F, 11.0F, 16.0F, 21.0F, 26.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 2.0F, 7.0F, 12.0F, 17.0F, 22.0F, 27.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 3.0F, 8.0F, 13.0F, 18.0F, 23.0F, 28.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 4.0F, 9.0F, 14.0F, 19.0F,
                24.0F, 29.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 5.0F, 10.0F,
                15.0F, 20.0F, 25.0F, 30.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 6.0F, 11.0F, 16.0F,
                21.0F, 26.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 7.0F,
                12.0F, 17.0F, 22.0F, 27.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                3.0F, 8.0F, 13.0F, 18.0F, 23.0F, 28.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 4.0F, 9.0F, 14.0F, 19.0F, 24.0F, 29.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 5.0F, 10.0F, 15.0F, 20.0F, 25.0F, 30.0F).reshape(2, 5, 2, 3, 2)
        val d2 = tensorOf(
                1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                3.0F, 0.0F, 0.0F, 0.0F, 0.0F, 4.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                5.0F, 0.0F, 0.0F, 0.0F, 0.0F, 6.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 3.0F, 0.0F, 0.0F, 0.0F, 0.0F, 4.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 0.0F, 6.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 3.0F, 0.0F, 0.0F, 0.0F, 0.0F, 4.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 0.0F, 6.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 3.0F, 0.0F, 0.0F, 0.0F, 0.0F, 4.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 0.0F, 6.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 2.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 3.0F, 0.0F, 0.0F, 0.0F, 0.0F, 4.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 5.0F, 0.0F, 0.0F, 0.0F, 0.0F, 6.0F,
                7.0F, 0.0F, 0.0F, 0.0F, 0.0F, 8.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                9.0F, 0.0F, 0.0F, 0.0F, 0.0F, 10.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                11.0F, 0.0F, 0.0F, 0.0F, 0.0F, 12.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 7.0F, 0.0F, 0.0F, 0.0F, 0.0F, 8.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 9.0F, 0.0F, 0.0F, 0.0F, 0.0F, 10.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 11.0F, 0.0F, 0.0F, 0.0F, 0.0F, 12.0F, 0.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 7.0F, 0.0F, 0.0F, 0.0F, 0.0F, 8.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 9.0F, 0.0F, 0.0F, 0.0F, 0.0F, 10.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 11.0F, 0.0F, 0.0F, 0.0F, 0.0F, 12.0F, 0.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 7.0F, 0.0F, 0.0F, 0.0F, 0.0F, 8.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 9.0F, 0.0F, 0.0F, 0.0F, 0.0F, 10.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 11.0F, 0.0F, 0.0F, 0.0F, 0.0F, 12.0F, 0.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 7.0F, 0.0F, 0.0F, 0.0F, 0.0F, 8.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 9.0F, 0.0F, 0.0F, 0.0F, 0.0F, 10.0F,
                0.0F, 0.0F, 0.0F, 0.0F, 11.0F, 0.0F, 0.0F, 0.0F, 0.0F, 12.0F).reshape(2, 5, 3, 2, 5)
        primalAndForwardDerivative(x, y, ::f) shouldBeExactly Pair(expectedPrimal, Pair(d1, d2))
        primalAndReverseDerivativeTransposed(x, y, ::f) shouldBeExactly Pair(expectedPrimal, Pair(d1, d2))
        /*
        primalAndForwardDerivative(x, y) { xx: DTensor, yy: DTensor -> primalAndForwardDerivative(xx, yy, ::f).second }
        primalAndForwardDerivative(x, y) { xx: DTensor, yy: DTensor -> primalAndForwardDerivative(xx, yy, ::f).third }
        primalAndForwardDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndTransposedReverseDerivative(xx, yy, ::f).second })
        primalAndForwardDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndTransposedReverseDerivative(xx, yy, ::f).third })
        primalAndTransposedReverseDerivative(x, y, ::f)
        primalAndTransposedReverseDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndForwardDerivative(xx, yy, ::f).second })
        primalAndTransposedReverseDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndForwardDerivative(xx, yy, ::f).third })
        primalAndTransposedReverseDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndTransposedReverseDerivative(xx, yy, ::f).second })
        primalAndTransposedReverseDerivative(x, y, { xx: DTensor, yy: DTensor -> primalAndTransposedReverseDerivative(xx, yy, ::f).third })
        */
        // TODO https://github.com/facebookincubator/diffkt/issues/99: should test the values of the derivatives
    }

    /**
     * Test fix for https://github.com/facebookresearch/diffkt/issues/73
     */
    @Test fun `regression 73`() {
        fun f(x: DTensor) : DTensor {
            val c = tensorOf(1f, 2f, 3f)
            val y = c.innerProduct(Shape(3), x.pow(2f))
            return y
        }

        val x = tensorOf(1f, 2f, 3f)
        val (_, _) = primalAndReverseDerivative(x, ::f)
    }
}

fun main() {
    InnerProductTest().`regression 73`()
}