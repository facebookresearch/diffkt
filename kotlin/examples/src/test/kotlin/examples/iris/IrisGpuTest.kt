/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.iris

import examples.api.Learner
import examples.api.SimpleDataIterator
import io.kotest.core.annotation.Tags
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeGreaterThan
import io.kotest.matchers.floats.shouldBeLessThan
import org.diffkt.Device
import org.diffkt.FloatScalar
import org.diffkt.basePrimal
import org.diffkt.model.FixedLearningRateOptimizer
import kotlin.random.Random

@Tags("Gpu")
class IrisGpuTest : AnnotationSpec() {
    @Test @Ignore
    fun testIris() {
        val initialModel = IrisModel(Random(12345)).gpu()
        val (features, labels) = loadData()
        val initialPrediction = initialModel.predict(features.gpu())
        val initialLoss = loss(initialPrediction, labels.gpu()).basePrimal().cpu()
        (initialLoss as FloatScalar).value shouldBeGreaterThan 1F

        val batchedData = SimpleDataIterator(features, labels, features.shape[0])
        val optimizer = FixedLearningRateOptimizer<IrisModel>(alpha)
        val learner = Learner(batchedData, ::loss, optimizer)
        val trainedModel = learner.train(initialModel, epochs = 400, device = Device.GPU)
        val finalPrediction = trainedModel.predict(features.gpu())
        val finalLoss = loss(finalPrediction, labels.gpu()).basePrimal().cpu()
        (finalLoss as FloatScalar).value shouldBeLessThan 1F
    }
}