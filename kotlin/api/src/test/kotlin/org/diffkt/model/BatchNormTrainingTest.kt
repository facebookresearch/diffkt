/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import testutils.floats
import testutils.shouldBeNear

class BatchNormTrainingTest : AnnotationSpec() {
    @Test
    fun testForwardTrainAndEvaluation() {
        val values = BatchNorm2dTestValues.BothModes
        val bn = values.bn

        // check for expected result during training
        val result0 = bn(values.input)
        result0.shouldBeNear(values.expectedTrainResult, values.epsilon)
        val (mean, variance) = bn.stats
        mean.shouldBeNear(values.expectedRunningMean, values.epsilon)
        variance.shouldBeNear(values.expectedRunningVariance, values.epsilon)

        // check for expected result during inference
        val frozen = bn.inferenceMode
        val result1 = frozen(values.testInput)
        result1.shouldBeNear(values.expectedTestResult, values.epsilon)
    }

    object BatchNorm2dTestValues {
        object BothModes {
            val epsilon = 8e-5f
            val bn get() = BatchNormTraining(3)
            val input = // Make a tensor where 1, 2, 3, 4 is the first image, 5, 6, 7, 8 is the second, etc
                run {
                    val ch1 = FloatTensor(Shape(1, 2, 2, 1), floats(4))
                    val ch2 = FloatTensor(Shape(1, 2, 2, 1), floats(4, start = 5))
                    val ch3 = FloatTensor(Shape(1, 2, 2, 1), floats(4, start = 9))
                    ch1.concat(ch2, axis = 3).concat(ch3, axis = 3)
                }
            val expectedTrainResult = FloatTensor(Shape(1, 2, 2, 3), floatArrayOf(
                -1.3416355F, -1.34164F, -1.3416443F,
                -0.44721186F, -0.44721365F, -0.44721508F,
                0.44721174F, 0.4472127F, 0.44721413F,
                1.3416355F, 1.341639F, 1.3416424F))

            private val axes = intArrayOf(0, 2, 1, 3)
            val testInput = input.transpose(axes)
            val expectedTestResult = expectedTrainResult.transpose(axes)

            val expectedRunningMean = tensorOf(2.5f, 6.5f, 10.5f)
            val expectedRunningVariance = FloatTensor(Shape(3)) { 1.25f }
        }
    }
}
