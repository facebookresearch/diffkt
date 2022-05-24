/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.french

import org.diffkt.data.loaders.french.French2EnglishDataLoader
import org.diffkt.model.SGDOptimizer
import kotlin.random.Random


fun main() {
    // Get data ready
    val dataLoader = French2EnglishDataLoader
    val filteredSentences = dataLoader.loadFilteredSentences()
    // Get our vocabularies built
    val (englishVocab, frenchVocab) = dataLoader.createVocabularies(filteredSentences)
    // Create the model
    val french2English = French2English(englishVocab, frenchVocab, random = Random(15312))
    // Create an optimizer
    val optimizer = SGDOptimizer<French2English>()
    // Train the model
    val t1 = System.currentTimeMillis()
    val trainedFrench2English = french2English.train(filteredSentences, 2000, printEvery = 100, optimizer = optimizer)
    val t2 = System.currentTimeMillis()
    println("training: ${(t2 - t1)/1000f}")

}