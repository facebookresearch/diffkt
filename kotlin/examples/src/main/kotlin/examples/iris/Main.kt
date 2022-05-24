/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.iris

import examples.api.Learner
import examples.api.SimpleDataIterator
import org.diffkt.model.FixedLearningRateOptimizer
import kotlin.random.Random

const val alpha = 0.015f

fun main() {
    val initialModel = IrisModel(Random(12345))
    val (features, labels) = loadData()
    val batchedData = SimpleDataIterator(features, labels, features.shape[0])

    val optimizer = FixedLearningRateOptimizer<IrisModel>(alpha)
    val learner = Learner(batchedData, ::loss, optimizer, useJit = true)
    /* val trainedModel = */ learner.train(initialModel, epochs = 80000, printProgress = false)
    learner.dumpTimes()
}
