/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.data.loaders.cifar

import org.diffkt.*

typealias CifarData = List<Pair<FloatTensor, FloatTensor>>

object Loader {
    fun loadTest(): CifarData {
        val test = FileReader.testSubset()
        return extract(test)
    }

    fun loadTrain(): CifarData {
        val datas = mutableListOf<Pair<FloatTensor, FloatTensor>>()
        for (set in 1 until 6) {
            val train = FileReader.exampleSubset(set)
            val ex = extract(train)
            datas.addAll(ex)
        }
        return datas
    }

    private fun extract(images: Array<ByteArray>): CifarData {
        val featureLabelPairs = images.map { image ->
            val label = FloatScalar(image[0].toFloat())
            val data = FloatArray(3 * 32 * 32) {
                image[it + 1].toUByte().toFloat() / 255f
            }
            val feature = FloatTensor(Shape(3, 32, 32), data)

            // Transpose to H x W x C
            val transposed = feature.transpose(intArrayOf(1, 2, 0)) as FloatTensor
            Pair(transposed, label)
        }
        return featureLabelPairs
    }
}
