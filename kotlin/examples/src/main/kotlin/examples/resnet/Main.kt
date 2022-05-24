/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.resnet

import examples.api.*
import org.diffkt.*
import org.diffkt.data.loaders.cifar.Loader
import org.diffkt.model.SGDOptimizer
import kotlin.random.Random

fun run(benchmark: Boolean = false, maxIters: Int? = null) {
    val random = Random(1234567)

    val t1 = System.nanoTime()

    // Functional version of the image normalizer transformer which works on batches
    fun normalize(batchedImages: DTensor): DTensor {
        require(batchedImages.rank == 4)
        val channelMeans = tensorOf(0.4914f, 0.4822f, 0.4465f)
        val channelStdevs = tensorOf(0.2023f, 0.1994f, 0.2010f)
        return (batchedImages - channelMeans) / channelStdevs
    }

    val trainData = Loader.loadTrain()
    val testData = Loader.loadTest()

    val trainDataIterator = TransformingDataIterator(
        trainData, { squareImageRandomFlip(randomSquareCropSameSize(it, 4, random), random = random) },
        ::normalize,
        batchSize = 128)
    /*val testDataIterator =*/ TransformingDataIterator(testData, batchTransform = ::normalize)
    val t2 = System.nanoTime()

    // Define the ResNet
    val optim = SGDOptimizer<ResNet>(0.1f, momentum = 0.9f, weightDecay = 0.00001f)
    val resnet = ResNet(listOf(2, 2, 2, 2), random = random)
    val t3 = System.nanoTime()

    // Train and test ResNet
    val learner = Learner(trainDataIterator, ::crossEntropyLoss, optim)
    /*val trainedResnet =*/ learner.train(
        resnet, epochs = 5,
        printProgress = !benchmark,
        maxIters = maxIters,
        printProgressFrequently = !benchmark)
    val t4 = System.nanoTime()

    fun elapsed(start: Long, end: Long): String {
        return "${(end - start) / 1e9f} sec"
    }

    if (benchmark) {
        println("resnet $maxIters total iterations")
        println("total:        ${elapsed(t1, t4)}")
        println("training:     ${elapsed(t3, t4)}")
        println("data loading: ${elapsed(t1, t2)}")
        return
    }
}

fun main(args: Array<String>) {
    // Special secret case for benchmarking
    // Usage: ./gradlew :examples:run -Ppackage=resnet --args="bench 2"
    if (args.isNotEmpty() && (args[0] == "bench" || args[0] == "b")) {
        val n = if (args.size > 1) args[1].toIntOrNull() else null
        run(benchmark = true, maxIters = n ?: 3)
    } else {
        run()
    }
}
