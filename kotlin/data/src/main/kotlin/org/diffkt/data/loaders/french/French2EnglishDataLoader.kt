/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.data.loaders.french

import org.diffkt.IntTensor
import org.diffkt.Shape
import org.diffkt.data.loaders.DATASET_DIR
import org.diffkt.data.loaders.downloadZip
import java.nio.file.Files
import java.nio.file.Paths
import java.text.Normalizer


typealias EngFrenchPair = Pair<String, String>

object French2EnglishDataLoader {

    private val url = "http://download.pytorch.org/tutorial/data.zip"
    private val dataDirectory = "$DATASET_DIR/french2english/"

    // Filter to reduce the number of sentence types involved in training
    // will only train on sentences that start with these prefixes in english
    private val sentenceFilters = listOf("i am ", "i m ",
        "you are", "you re ",
        "she is", "she s ",
        "he is", "he s ",
        "they are", "they re ",
        "we are", "we re "
    )

    // Loads entire file into memory.
    private fun load(): List<String> {
        if (!Files.exists(Paths.get(dataDirectory + "data/eng-fra.txt"))) {
            downloadZip(url, dataDirectory)
        }
        return Paths.get("$dataDirectory/data/eng-fra.txt").toFile().readLines(Charsets.UTF_8)
    }

    fun loadFilteredSentences(): List<EngFrenchPair> {
        val rawLines = load().map { it.split("\t") }
        // Normalize the lines to remove extra spaces, accents and capital letters
        val cleanLines = rawLines.map { line -> Pair(normalize(line[0]), normalize(unicodeify(line[1]))) }
        // Remove sentences that don't start with accepted sentence types from filter
        val linesSentenceFilter = filterBySentenceStart(cleanLines, sentenceFilters)
        // Limit sentence length to 10 words.
        return filterBySentenceLength(linesSentenceFilter, 10)
    }

    fun createVocabularies(sentences: List<EngFrenchPair>): Pair<Vocabulary, Vocabulary> {
        val englishVocab = Vocabulary()
        val frenchVocab = Vocabulary()
        for (example in sentences) {
            val english = example.first
            val french = example.second
            englishVocab.addSentence(english.split(" "))
            frenchVocab.addSentence(french.split(" "))
        }
        return Pair(englishVocab, frenchVocab)
    }


    private fun filterBySentenceStart(input: List<EngFrenchPair>, starts: List<String>): List<EngFrenchPair> {
        return input.filter { l -> starts.fold(false) {b, s -> b || l.first.startsWith(s)} }
    }

    private fun filterBySentenceLength(input: List<EngFrenchPair>, length: Int): List<EngFrenchPair> {
        return input.filter { l -> l.first.split(" ").size < length && l.second.split(" ").size < length }
    }

    private fun normalize(w: String): String {
        // Replace any non letter character with a space, convert to lowercase, remove spaces from beginning or end of string
        var final = w.replace(Regex("[^a-zA-Z.!? ]+"), " ").toLowerCase().trim()
        // Add a space before any punctuation
        listOf(".", "?", "!").forEach { final = final.replace(it, " $it") }
        // Replace any sequence of spaces with a single space
        final = final.replace(Regex("[ ]+"), " ")
        return final
    }

    private fun unicodeify(s: String): String {
        // Convert accent marks into letters or remove them if they cannot be converted
        val string = Normalizer.normalize(s, Normalizer.Form.NFD)
        return Regex("\\p{InCombiningDiacriticalMarks}+").replace(string, "")
    }
}