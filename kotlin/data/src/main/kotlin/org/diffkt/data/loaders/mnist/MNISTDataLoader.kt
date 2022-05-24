/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.data.loaders.mnist

import org.diffkt.data.Data
import org.diffkt.*
import org.diffkt.data.loaders.DATASET_DIR
import org.diffkt.data.loaders.downloadFile
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.file.Files
import java.nio.file.Paths
import java.util.zip.GZIPInputStream

const val ImageHW = 28
const val NumClasses = 10
const val NumTrain = 6000
const val NumTest = 1000
const val NumExamples = 70000
const val NumChannels = 1

class MNISTDataLoader(val oneHotLabels: Boolean = true) {
    private val numClasses = NumClasses
    val dataDirectory = "$DATASET_DIR/mnist/"
    val url = "http://yann.lecun.com/exdb/mnist/"

    val data = load()
    fun getAllExamples(): Data { return data.first }
    fun getTrainExamples(): Data { return data.second }
    fun getTestExamples(): Data { return data.third }

    private fun load(printout: Boolean = false): Triple<Data, // all
                                                Data, // training
                                                Data> // testing
    {
        val files = listOf(
            "train-labels-idx1-ubyte.gz",
            "train-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz"
        )
        files.forEach {
            if (!Files.exists(Paths.get(dataDirectory + it)))
                downloadFile(url + it, dataDirectory, it)
        }

        val paths = files.map { dataDirectory + it }
        val trainImages = loadImageSet(paths[1])
        val trainLabels = loadLabelSet(paths[0])
        val testImages = loadImageSet(paths[3])
        val testLabels = loadLabelSet(paths[2])
        val allImages = trainImages.concat(testImages)
        val allLabels = trainLabels.concat(testLabels)

        val allExamples = Data(allImages, allLabels)
        val trainExamples = Data(trainImages, trainLabels)
        val testExamples = Data(testImages, testLabels)

        if (printout) {
            printDatasetSize(trainExamples, "train")
            printDatasetSize(testExamples, "test")
        }

        return Triple(allExamples, trainExamples, testExamples)
    }

    private fun loadImageSet(path: String): FloatTensor {
        val file = Paths.get(path).toFile()
        val stream = GZIPInputStream(FileInputStream(file))
        val buffer = ByteArray(16384)
        stream.read(buffer, 0, 16)
        val numImages = ByteBuffer.wrap(buffer.slice(4 until 8).toByteArray()).int
        val numRows = ByteBuffer.wrap(buffer.slice(8 until 12).toByteArray()).int
        val numColumns = ByteBuffer.wrap(buffer.slice(12 until 16).toByteArray()).int
        val data = FloatArray(numImages * numRows * numColumns)
        var off = 0
        var bytesIn: Int
        while (run { bytesIn = stream.read(buffer, 0, buffer.size); bytesIn >= 0 }) {
            var i = 0
            while (i < bytesIn) {
                val a = buffer[i].toInt()
                data[off] = (a and 0xFF) / 255f
                i += 1
                off += 1
            }
        }
        return FloatTensor(Shape(numImages, numRows, numColumns), data)
    }

    private fun oneHot(cls: Int): FloatArray {
        val a = FloatArray(numClasses) { 0f }
        a[cls] = 1f
        return a
    }

    private fun loadLabelSet(path: String): FloatTensor {
        val file = Paths.get(path).toFile()
        val stream = GZIPInputStream(FileInputStream(file))
        val buffer = ByteArray(8192) { Byte.MIN_VALUE }
        stream.read(buffer, 0, 8)
        val numLabels = ByteBuffer.wrap(buffer.slice(4 until 8).toByteArray()).int
        val data = if (oneHotLabels) FloatArray(numLabels * numClasses) else FloatArray(numLabels)
        var pos = 0
        var bytesIn: Int
        val labelSize = if (oneHotLabels) numClasses else 1
        while (run { bytesIn = stream.read(buffer, 0, buffer.size); bytesIn >= 0 }) {
            var i = 0
            while (i < bytesIn) {
                val label = buffer[i].toInt()
                if (oneHotLabels)
                    oneHot(label).copyInto(data, pos, 0, numClasses)
                else
                    data[pos] = label.toFloat()
                i += 1
                pos += labelSize
            }
        }
        return if (oneHotLabels)
            FloatTensor(Shape(numLabels, numClasses), data)
        else
            FloatTensor(Shape(numLabels), data)
    }

    private fun printDatasetSize(examples: Data, purpose: String) {
        println("$purpose set: ${examples.features.shape.first}" +
                " examples, dimensions ${examples.features.shape.drop(1)}, from $numClasses classes")
    }
}
