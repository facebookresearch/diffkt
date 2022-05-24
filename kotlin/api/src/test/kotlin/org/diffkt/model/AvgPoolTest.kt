/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.data.forAll
import io.kotest.data.headers
import io.kotest.data.row
import io.kotest.data.table
import io.kotest.matchers.string.shouldContain
import org.diffkt.*
import testutils.floats
import kotlin.test.assertEquals

class AvgPoolTest : AnnotationSpec() {
    @Test
    suspend fun avgPool() {
        forAll(table(
            headers("Input", "Expected Out", "Expected Grad"),
            row(
                AvgPoolTestValues.AvgPool.inputTensor,
                AvgPoolTestValues.AvgPool.expectedOutput,
                AvgPoolTestValues.AvgPool.expectedGradient
            ),
            row(
                AvgPoolTestValues.AvgPool2.inputTensor,
                AvgPoolTestValues.AvgPool2.expectedOutput,
                AvgPoolTestValues.AvgPool2.expectedGradient
            )
        )) { inputTensor, expectedOutput, expectedGradient ->
            val (output, grad) = primalAndVjp(inputTensor, { it }) {
                avgPool(it, 2, 2)
            }
            if (expectedOutput != output)
                if (!expectedOutput.equals(output))
                    assertEquals(expectedOutput, output) // TODO: delete this if statement

            assertEquals(expectedOutput, output)

            // note: non-unitary seed to verify correct reverse indexing in gradient
            // nothing special about output as gradient seed, it's just the right shape
            assertEquals(expectedGradient, grad)
        }
    }

    @Test
    fun nonDivisibleHeight() {
        val shape = Shape(1, 5, 4, 1)
        val tensor = FloatTensor(shape, floats(shape.product()))
        val e = shouldThrow<IllegalArgumentException> { avgPool(tensor, 2, 2) }
        e.message shouldContain "height"
    }

    @Test
    fun nonDivisibleWidth() {
        val shape = Shape(1, 4, 5, 1)
        val tensor = FloatTensor(shape, floats(shape.product()))
        val e = shouldThrow<IllegalArgumentException> { avgPool(tensor, 2, 2) }
        e.message shouldContain "width"
    }
}

object AvgPoolTestValues {
    object AvgPool {
        val inputTensor get() = FloatTensor(Shape(2, 6, 6, 3), floats(108) + floats(108))

        val expectedGradient = FloatTensor(
            Shape(2, 6, 6, 3),
                2.875f, 3.125f, 3.375f, 2.875f, 3.125f, 3.375f, 4.375f, 4.625f, 4.875f, 4.375f, 4.625f, 4.875f,
                5.875f, 6.125f, 6.375f, 5.875f, 6.125f, 6.375f, 2.875f, 3.125f, 3.375f, 2.875f, 3.125f, 3.375f,
                4.375f, 4.625f, 4.875f, 4.375f, 4.625f, 4.875f, 5.875f, 6.125f, 6.375f, 5.875f, 6.125f, 6.375f,
                11.875f, 12.125f, 12.375f, 11.875f, 12.125f, 12.375f, 13.375f, 13.625f, 13.875f, 13.375f,
                13.625f, 13.875f, 14.875f, 15.125f, 15.375f, 14.875f, 15.125f, 15.375f, 11.875f, 12.125f,
                12.375f, 11.875f, 12.125f, 12.375f, 13.375f, 13.625f, 13.875f, 13.375f, 13.625f, 13.875f,
                14.875f, 15.125f, 15.375f, 14.875f, 15.125f, 15.375f, 20.875f, 21.125f, 21.375f, 20.875f,
                21.125f, 21.375f, 22.375f, 22.625f, 22.875f, 22.375f, 22.625f, 22.875f, 23.875f, 24.125f,
                24.375f, 23.875f, 24.125f, 24.375f, 20.875f, 21.125f, 21.375f, 20.875f, 21.125f, 21.375f,
                22.375f, 22.625f, 22.875f, 22.375f, 22.625f, 22.875f, 23.875f, 24.125f, 24.375f, 23.875f,
                24.125f, 24.375f,

                2.875f, 3.125f, 3.375f, 2.875f, 3.125f, 3.375f, 4.375f, 4.625f, 4.875f, 4.375f, 4.625f, 4.875f,
                5.875f, 6.125f, 6.375f, 5.875f, 6.125f, 6.375f, 2.875f, 3.125f, 3.375f, 2.875f, 3.125f, 3.375f,
                4.375f, 4.625f, 4.875f, 4.375f, 4.625f, 4.875f, 5.875f, 6.125f, 6.375f, 5.875f, 6.125f, 6.375f,
                11.875f, 12.125f, 12.375f, 11.875f, 12.125f, 12.375f, 13.375f, 13.625f, 13.875f, 13.375f,
                13.625f, 13.875f, 14.875f, 15.125f, 15.375f, 14.875f, 15.125f, 15.375f, 11.875f, 12.125f,
                12.375f, 11.875f, 12.125f, 12.375f, 13.375f, 13.625f, 13.875f, 13.375f, 13.625f, 13.875f,
                14.875f, 15.125f, 15.375f, 14.875f, 15.125f, 15.375f, 20.875f, 21.125f, 21.375f, 20.875f,
                21.125f, 21.375f, 22.375f, 22.625f, 22.875f, 22.375f, 22.625f, 22.875f, 23.875f, 24.125f,
                24.375f, 23.875f, 24.125f, 24.375f, 20.875f, 21.125f, 21.375f, 20.875f, 21.125f, 21.375f,
                22.375f, 22.625f, 22.875f, 22.375f, 22.625f, 22.875f, 23.875f, 24.125f, 24.375f, 23.875f,
                24.125f, 24.375f
        )

        val expectedOutput = FloatTensor(
            Shape(2, 3, 3, 3),
                11.5f, 12.5f, 13.5f, 17.5f, 18.5f, 19.5f, 23.5f, 24.5f, 25.5f,
                47.5f, 48.5f, 49.5f, 53.5f, 54.5f, 55.5f, 59.5f, 60.5f, 61.5f,
                83.5f, 84.5f, 85.5f, 89.5f, 90.5f, 91.5f, 95.5f, 96.5f, 97.5f,

                11.5f, 12.5f, 13.5f, 17.5f, 18.5f, 19.5f, 23.5f, 24.5f, 25.5f,
                47.5f, 48.5f, 49.5f, 53.5f, 54.5f, 55.5f, 59.5f, 60.5f, 61.5f,
                83.5f, 84.5f, 85.5f, 89.5f, 90.5f, 91.5f, 95.5f, 96.5f, 97.5f
        )
    }

    object AvgPool2 {
        val inputTensor get() = FloatTensor(
            Shape(2, 4, 4, 2),
                8f, 1f, 5f, 2f, 1f, 4f, 2f, 6f,
                1f, 4f, 2f, 3f, 4f, 0f, 3f, -1f,
                4f, 8f, 6f, 5f, 3f, 3f, 0f, 0f,
                0f, 1f, -1f, 2f, 5f, 5f, 0f, 0f,

                8f, 1f, 5f, 2f, 1f, 4f, 2f, 6f,
                1f, 4f, 2f, 3f, 4f, 0f, 3f, -1f,
                4f, 8f, 6f, 5f, 3f, 3f, 0f, 0f,
                0f, 1f, -1f, 2f, 5f, 5f, 0f, 0f
        )

        val expectedGradient = FloatTensor(
            Shape(2, 4, 4, 2),
                1.0f, 0.625f, 1.0f, 0.625f, 0.625f, 0.5625f,
                0.625f, 0.5625f, 1.0f, 0.625f, 1.0f, 0.625f,
                0.625f, 0.5625f, 0.625f, 0.5625f, 0.5625f, 1.0f,
                0.5625f, 1.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5625f,
                1.0f, 0.5625f, 1.0f, 0.5f, 0.5f, 0.5f, 0.5f,

                1.0f, 0.625f, 1.0f, 0.625f, 0.625f, 0.5625f,
                0.625f, 0.5625f, 1.0f, 0.625f, 1.0f, 0.625f,
                0.625f, 0.5625f, 0.625f, 0.5625f, 0.5625f, 1.0f,
                0.5625f, 1.0f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5625f,
                1.0f, 0.5625f, 1.0f, 0.5f, 0.5f, 0.5f, 0.5f
        )

        val expectedOutput = FloatTensor(
            Shape(2, 2, 2, 2),
                4.0f, 2.5f,
                2.5f, 2.25f,
                2.25f, 4.0f,
                2.0f, 2.0f,

                4.0f, 2.5f,
                2.5f, 2.25f,
                2.25f, 4.0f,
                2.0f, 2.0f
        )
    }
}
