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

data class NSchoolsAttrs(val states: List<Int>, val districts: List<Int>, val types: List<Int>)
data class NSchoolsData(val attrs: NSchoolsAttrs, val y: FloatTensor, val sigma: FloatTensor)

object NSchoolsLoader {
    val csvFileTrain = "NSchools.csv"
    val csvSplitBy = ", "

    fun load(): Pair<NSchoolsData, NSchoolsData> {
        val stream: InputStream = Thread.currentThread().contextClassLoader.getResourceAsStream(csvFileTrain)!!
        val br = BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8))

        val trainStates = br.readLine().split(csvSplitBy).map { it.toInt() }
        val trainDistricts = br.readLine().split(csvSplitBy).map { it.toInt() }
        val trainTypes = br.readLine().split(csvSplitBy).map { it.toInt() }
        val trainYs = br.readLine().split(csvSplitBy).map { it.toFloat() }
        val trainSigmas = br.readLine().split(csvSplitBy).map { it.toFloat() }
        val trainData = NSchoolsData(
            NSchoolsAttrs(trainStates, trainDistricts, trainTypes),
            FloatTensor(Shape(1000), trainYs.toFloatArray()),
            FloatTensor(Shape(1000), trainSigmas.toFloatArray()),
        )

        val testStates = br.readLine().split(csvSplitBy).map { it.toInt() }
        val testDistricts = br.readLine().split(csvSplitBy).map { it.toInt() }
        val testTypes = br.readLine().split(csvSplitBy).map { it.toInt() }
        val testYs = br.readLine().split(csvSplitBy).map { it.toFloat() }
        val testSigmas = br.readLine().split(csvSplitBy).map { it.toFloat() }
        val testData = NSchoolsData(
            NSchoolsAttrs(testStates, testDistricts, testTypes),
            FloatTensor(Shape(1000), testYs.toFloatArray()),
            FloatTensor(Shape(1000), testSigmas.toFloatArray()),
        )

        return Pair(trainData, testData)
    }
}