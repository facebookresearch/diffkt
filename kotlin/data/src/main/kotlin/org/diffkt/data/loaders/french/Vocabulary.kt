/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.data.loaders.french

import org.diffkt.IntTensor
import org.diffkt.Shape

class Vocabulary {
    private var idx = 2
    val word2Idx = mutableMapOf("SOS" to 0, "EOS" to 1)
    val idx2Word = mutableMapOf(0 to "SOS", 1 to "EOS")
    val size get() = idx

    fun addWord(w: String) {
        if (!word2Idx.containsKey(w)) { word2Idx.put(w, idx); idx2Word.put(idx, w); idx += 1 }
    }

    fun addSentence(s: List<String>) {
        for (w in s) { addWord(w) }
    }

    // converts sentence to int tensor with appended EOS token
    fun sentenceToTensor(sentence: String): IntTensor {
        val res = sentence.split(" ").map { word -> word2Idx[word]!! } + listOf(1)
        return IntTensor(Shape(res.size), res.toIntArray())
    }
}
