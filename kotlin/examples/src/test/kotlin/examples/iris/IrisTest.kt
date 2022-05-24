/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.iris

import examples.api.Learner
import examples.api.SimpleDataIterator
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeGreaterThan
import io.kotest.matchers.floats.shouldBeLessThan
import org.diffkt.model.FixedLearningRateOptimizer
import kotlin.random.Random
import testutils.*

class IrisTest : AnnotationSpec() {
    @Test
    fun testIris() {
        val initialModel = IrisModel(Random(12345))
        val (features, labels) = loadData()
        val initialPrediction = initialModel.predict(features)
        loss(initialPrediction, labels).value shouldBeGreaterThan 1F

        val batchedData = SimpleDataIterator(features, labels, features.shape[0])
        val optimizer = FixedLearningRateOptimizer<IrisModel>(alpha)
        val learner = Learner(batchedData, ::loss, optimizer, useJit = true)
        val trainedModel = learner.train(initialModel, epochs = 400)
        val finalPrediction = trainedModel.predict(features)
        loss(finalPrediction, labels).value shouldBeLessThan 1F
    }
}
