/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeExactly
import io.kotest.matchers.ints.shouldBeExactly
import org.diffkt.*
import testutils.shouldBeExactly

class RMSpropTest : AnnotationSpec() {
    class InstrumentedRMSprop<T : Model<T>>(
        alpha: Float = 0.005f,
        beta: Float = 0.9f,
        ): RMSpropOptimizer<T>(alpha, beta) {
        val nextParameterValue get() = nextParameter
        val meanSquaresValue get() = meanSquares

        var afterFitCalls: Int = 0
        override fun afterFit() {
            afterFitCalls++
            super.afterFit()
        }

        var tensorTrainingStepCalls: Int = 0
        override fun tensorTrainingStep(tensor: DTensor, gradient: DTensor): DTensor {
            tensorTrainingStepCalls++
            return super.tensorTrainingStep(tensor, gradient)
        }
    }

    class SampleModel(val t: AffineTransform): Model<SampleModel>() {
        override val layers: List<Layer<*>>
            get() = listOf<Layer<*>>(t)

        override fun withLayers(newLayers: List<Layer<*>>): SampleModel {
            require(newLayers.size == 1)
            return SampleModel(newLayers[0] as AffineTransform)
        }

        override fun hashCode(): Int = combineHash("SampleModel", t)
        override fun equals(other: Any?): Boolean = other is SampleModel && other.t == t
    }

    @Test
    fun checkRMSpropBehavior() {
        val opt = InstrumentedRMSprop<SampleModel>()
        opt.alpha shouldBeExactly  0.005f
        opt.beta shouldBeExactly 0.9f
        opt.nextParameterValue shouldBeExactly 0
        opt.meanSquaresValue.size shouldBeExactly 0
        opt.afterFitCalls shouldBeExactly 0
        opt.tensorTrainingStepCalls shouldBeExactly 0

        val m = FloatScalar(1f)
        val b = FloatScalar(10f)
        val model = SampleModel(AffineTransform(TrainableTensor(m), TrainableTensor(b)))
        val tangent1 = TrainableComponent.Companion.Tangent( listOf(// for SampleModel
            TrainableComponent.Companion.Tangent( listOf( // for AffineTransform
                TrainableTensor.Companion.Tangent(FloatScalar(100f)), // for m
                TrainableTensor.Companion.Tangent(FloatScalar(1000f)), // for b
            ))))

        val model1 = opt.train(model, tangent1)
        model1.t.m.tensor shouldBeExactly FloatScalar(0.995f)
        model1.t.b.tensor shouldBeExactly FloatScalar(9.995f)

        opt.afterFitCalls shouldBeExactly 1
        opt.tensorTrainingStepCalls shouldBeExactly 2
        opt.nextParameterValue shouldBeExactly 0
        opt.meanSquaresValue.size shouldBeExactly 2

        opt.meanSquaresValue[0] shouldBeExactly FloatScalar(10000f)
        opt.meanSquaresValue[1] shouldBeExactly FloatScalar(1000000f)

        val tangent2 = TrainableComponent.Companion.Tangent( listOf(// for SampleModel
            TrainableComponent.Companion.Tangent( listOf( // for AffineTransform
                TrainableTensor.Companion.Tangent(FloatScalar(2f)), // for m
                TrainableTensor.Companion.Tangent(FloatScalar(3f)), // for b
            ))))

        val model2 = opt.train(model1, tangent2)
        model2.t.m.tensor shouldBeExactly FloatScalar(0.9948946f)
        model2.t.b.tensor shouldBeExactly FloatScalar(9.994984f)

        opt.afterFitCalls shouldBeExactly 2
        opt.tensorTrainingStepCalls shouldBeExactly 4
        opt.nextParameterValue shouldBeExactly 0
        opt.meanSquaresValue.size shouldBeExactly 2

        opt.meanSquaresValue[0] shouldBeExactly FloatScalar(9000.4f)
        opt.meanSquaresValue[1] shouldBeExactly FloatScalar(900000.9f)
    }
}