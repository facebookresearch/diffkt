/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.data.loaders.ppl

import org.diffkt.*
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets

class LogRegressionData(val xArray: FloatArray, val yArray: FloatArray, size: Shape) {
    val xTensor = FloatTensor(size, xArray)
    val yTensor = FloatTensor(size, yArray)
}

object LogRegressionLoader {
    val csvFileTrain = "logRegression.csv"
    val csvSplitBy = ", "

    fun load(): Pair<LogRegressionData, LogRegressionData> {
        val stream: InputStream = Thread.currentThread().contextClassLoader.getResourceAsStream(csvFileTrain)!!
        val br = BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8))
        val train_data = br.readLine().split(csvSplitBy).map { it.toFloat() }
        val train_labels = br.readLine().split(csvSplitBy).map { it.toFloat() }
        val train = LogRegressionData(train_data.toFloatArray(), train_labels.toFloatArray(), Shape(1000, 1))

        val test_data = br.readLine().split(csvSplitBy).map { it.toFloat() }
        val test_labels = br.readLine().split(csvSplitBy).map { it.toFloat() }
        val test = LogRegressionData(test_data.toFloatArray(), test_labels.toFloatArray(), Shape(1000, 1))
        return Pair(train, test)
    }
}
