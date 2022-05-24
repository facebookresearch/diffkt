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

class BatchNormTrainingV1Test : AnnotationSpec() {
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
            val bn get() = BatchNormTrainingV1(3)
            val input = // Make a tensor where 1, 2, 3, 4 is the first image, 5, 6, 7, 8 is the second, etc
                run {
                    val ch1 = FloatTensor(Shape(1, 2, 2, 1), floats(4))
                    val ch2 = FloatTensor(Shape(1, 2, 2, 1), floats(4, start = 5))
                    val ch3 = FloatTensor(Shape(1, 2, 2, 1), floats(4, start = 9))
                    ch1.concat(ch2, axis = 3).concat(ch3, axis = 3)
                }
            val expectedTrainResult = FloatTensor(Shape(1, 2, 2, 3), floatArrayOf(
                -1.3416355F, -1.3416355F, -1.3416355F,
                -0.44721183F, -0.44721183F, -0.44721183F,
                0.44721183F, 0.44721183F, 0.44721183F,
                1.3416355F, 1.3416355F, 1.3416355F))
            private val axes = intArrayOf(0, 2, 1, 3)
            val testInput = input.transpose(axes)
            val expectedTestResult = FloatTensor(Shape(1, 2, 2, 3), floatArrayOf(
                0.72618437f, 4.2118692f, 7.697554f,
                2.662676f, 6.148361f, 9.634046f,
                1.6944302f, 5.180115f, 8.6658f,
                3.6309218f, 7.1166067f, 10.602292f))
            val expectedRunningMean = tensorOf(0.2500f, 0.6500f, 1.0500f)
            val expectedRunningVariance = FloatTensor(Shape(3)) { 1.0667f }
        }
    }
}
