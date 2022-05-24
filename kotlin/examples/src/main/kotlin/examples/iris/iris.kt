/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.iris

import org.diffkt.data.loaders.iris.Iris
import org.diffkt.data.loaders.iris.IrisDataLoader
import org.diffkt.data.loaders.iris.IrisDataLoader.NUM_FEATURES
import org.diffkt.data.loaders.iris.IrisDataLoader.NUM_SPECIES
import org.diffkt.data.loaders.iris.Species
import org.diffkt.*

// returns the features and labels
fun loadData(): Pair<FloatTensor, FloatTensor> {
    val tests = IrisDataLoader.tests
    val nSamples = IrisDataLoader.tests.size
    val f1 = { i: Iris -> i.features }
    val featuresData = tests.map { it.first }.flatMap(f1).toFloatArray()
    val features = FloatTensor(Shape(nSamples, NUM_FEATURES), featuresData)
    val f2 = { i: Species -> i.oneHotLabel }
    val labelsData = tests.map { it.second }.flatMap(f2).toFloatArray()
    val labels = FloatTensor(Shape(nSamples, NUM_SPECIES), labelsData)
    return Pair(features, labels)
}

fun loss(prediction: DTensor, labels: DTensor): DScalar {
    return crossEntropyLossFromOneHot(prediction, labels)
}
