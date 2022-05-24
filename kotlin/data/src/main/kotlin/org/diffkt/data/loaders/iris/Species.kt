/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.data.loaders.iris

sealed class Species(val label: Int) {
    companion object SpeciesAnalysis {
        fun getHotLabel(label: Int): List<Float> = when (label) {
            0 -> listOf(1f, 0f, 0f)
            1 -> listOf(0f, 1f, 0f)
            2 -> listOf(0f, 0f, 1f)
            else -> throw IllegalArgumentException("Invalid label")
        }
        fun fromString(s: String): Species = when (s) {
            "Iris-setosa" -> Setosa
            "Iris-versicolor" -> Versicolor
            "Iris-virginica" -> Virginica
            else -> throw IllegalArgumentException("Invalid species")
        }
    }
    val oneHotLabel = getHotLabel(label)
}

object Setosa : Species(0)
object Versicolor : Species(1)
object Virginica : Species(2)
