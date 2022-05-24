/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.linreg

import examples.api.Learner
import examples.api.SimpleDataIterator
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.plusOrMinus
import io.kotest.matchers.shouldBe
import org.diffkt.*
import org.diffkt.model.FixedLearningRateOptimizer
import kotlin.random.Random
import testutils.*

class LinregTest : AnnotationSpec() {
    @Test
    fun linregTest() {
        val random = Random(1234567)
        val trueWeight = FloatScalar(random.nextFloat())
        val trueBias = FloatScalar(random.nextFloat())
        val linReg = LinearRegression(random)
        val features = FloatTensor(Shape(BATCH_SIZE)) { random.nextFloat() }
        val labels = (features * trueWeight + trueBias) as FloatTensor
        val optimizer = FixedLearningRateOptimizer<LinearRegression>(0.5F / BATCH_SIZE)
        val dataIterator = SimpleDataIterator(features, labels, batchSize = BATCH_SIZE)
        fun lossFunc(predictions: DTensor, labels: DTensor): DScalar {
            val diff = predictions - labels
            return (diff * diff).sum()
        }
        val learner = Learner(
            batchedData = dataIterator,
            lossFunc = ::lossFunc,
            optimizer = optimizer,
            useJit = true
        )
        val trainedLinreg = learner.train(linReg, 100, printProgress = false)
        lossFunc(trainedLinreg.predict(features), labels).value shouldBe (1.3274993E-7f plusOrMinus 5.0E-7f)
    }
}
