/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.data.loaders.iris

import org.diffkt.*
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.charset.StandardCharsets
import java.util.stream.Collectors

object IrisDataLoader {
    const val NUM_SPECIES = 3
    const val NUM_FEATURES = 4
    const val MIDDLE_DATA = 3

    // training data is collections pairs of Iris and Species objects, no tensors

    private fun fileData(): List<String> {
        // Based on https://stackoverflow.com/a/15749281
        val stream : InputStream = Thread.currentThread().contextClassLoader.getResourceAsStream("IrisData.csv")!!
        // Based on https://stackoverflow.com/a/29282588
        return BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8)).lines().collect(Collectors.toList())
    }

    private fun pairs(lines: List<String>): List<Pair<Iris, Species>> {
        return lines.map { line ->
            val fields = line.split(',')
            val iris = Iris(
                fields[0].toFloat(),
                fields[1].toFloat(),
                fields[2].toFloat(),
                fields[3].toFloat()
            )
            val species = Species.fromString(fields[4])
            Pair(iris, species)
        }
    }

    private fun assembleTensorData(lines: List<String>): Pair<FloatTensor, FloatTensor> {
        var featureData = floatArrayOf()
        var labelData = floatArrayOf()
        lines.map { line ->
            val fields = line.split(',')
            featureData = featureData + fields.take(4).map { it.toFloat() }
            labelData = labelData + Species.fromString(fields[4]).oneHotLabel
        }
        return Pair(FloatTensor(Shape(lines.size, NUM_FEATURES), featureData), FloatTensor(Shape(lines.size, NUM_SPECIES), labelData))
    }

    // For IrisLearner, feature and label tensors computed directly
    private val assembledData = assembleTensorData(fileData())
    val features = assembledData.first
    val labels = assembledData.second

    // For IrisWithClasses + similar, Pairs of Iris and Species
    private const val trainRatio = 0.90
    val examples = mutableListOf<Pair<Iris, Species>>()
    val tests = mutableListOf<Pair<Iris, Species>>()

    init {
        // evenly distribute kinds of Irises between the examples and tests lists
        pairs(fileData())
            .groupBy { it.second }.values
            .forEach { it.withIndex()
                .forEach { k ->
                    if (k.index < it.size * trainRatio) { examples.add(k.value) } else { tests.add(k.value) } } }
    }

    // Helper functions.
    fun List<Pair<Iris, Species>>.features() = this.map { it.first }

    @JvmName("IrisFeaturesToTensor")
    fun List<Iris>.toTensor() =
        FloatTensor(Shape(this.size, NUM_FEATURES), this.flatMap { i -> i.features }.toFloatArray())

    fun List<Pair<Iris, Species>>.labels() = this.map { it.second }

    @JvmName("IrisLabelsToTensor")
    fun List<Species>.toTensor() =
        FloatTensor(Shape(this.size), this.map { i -> i.label.toFloat() }.toFloatArray())

    fun List<Species>.toOneHotTensor() =
        FloatTensor(Shape(this.size, NUM_SPECIES), this.flatMap { i -> i.oneHotLabel }.toFloatArray())
}
