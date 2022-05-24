/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.data.loaders.cifar

import org.diffkt.data.loaders.DATASET_DIR
import org.diffkt.data.loaders.downloadFile
import java.io.*
import java.nio.file.Files
import java.nio.file.Paths
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream

/**
 * Reads CIFAR files from the diffkt datasets dir on disk. Downloads files if they do not exist.
 */
object FileReader {
    /**
     * The url stores a compressed directory that contains 8 files:
     * 5 of these files data_batch_x.bin are binary files storing a subset of the dataset intended to be used for training
     * 1 of these files test_batch.bin is a binary file storing a subset of the data intended to be used for testing
     * 1 of these files batches.meta.txt stores a list of the classes
     * 1 readme.html
     *
     * Each instance of a data set is a 32 by 32 pixel image, for a total size of 1024 pixels each.
     * Each pixel is comprised of three colors, red green and blue. The first 1024: red, second 1024: green, final 1024: blue
     * This results in a feature size of 1024 * 3 = 3072
     * each row of the data set includes both the features and the label for a total size of 3073 bytes per instance
     */
    private const val URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    private const val INSTANCE_BYTE_SIZE = 3073
    private const val TRAINING_SET_SIZE = 10000
    private val OUT_DIR = "$DATASET_DIR/cifar"

    val directory: String = run {
        val fileName = "cifar10.tar.gz"
        val path = Paths.get("$OUT_DIR/$fileName")
        if (!Files.exists(path)) {
            print("Dataset not found. Downloading...")
            downloadFile(URL, OUT_DIR, fileName)
            println(" done.")
        }
        extractTarGZ(path.toString())
        val directories = File(OUT_DIR).listFiles { file -> file.isDirectory }
        "$OUT_DIR/${directories!![0].name}"
    }

    private fun byteArrayForDataFile(dataFile: String): Array<ByteArray> {
        require(Files.exists(Paths.get(dataFile))) { "no $dataFile" }
        val bis = BufferedInputStream(FileInputStream(dataFile))
        val rows = Array(TRAINING_SET_SIZE) { ByteArray(0) }
        for (i in 0 until TRAINING_SET_SIZE) {
            rows[i] = ByteArray(INSTANCE_BYTE_SIZE)
            bis.read(rows[i], 0, INSTANCE_BYTE_SIZE)
        }
        bis.close()
        return rows
    }

    fun exampleSubset(index: Int): Array<ByteArray> {
        require(index in 1..5) { "Expected an index between 1 and 5, but received $index" }
        val dataFile = "$directory/data_batch_$index.bin"
        return byteArrayForDataFile(dataFile)
    }

    fun testSubset(): Array<ByteArray> {
        val dataFile = "$directory/test_batch.bin"
        return byteArrayForDataFile(dataFile)
    }

    private fun extractTarGZ(filePath: String): Boolean {
        val bufferSize = 1024
        val path = Paths.get(filePath)
        val directory = path.parent
        val fileStream = FileInputStream(filePath)
        val gzipIn = GzipCompressorInputStream(fileStream)
        val tarIn = TarArchiveInputStream(gzipIn)
        var entry = tarIn.nextTarEntry
        var stop = 0
        while (entry != null) {
            if (entry.isDirectory) {
                val f = File("$directory/${entry.name}")
                val created = f.mkdirs()
                // This file already exists, don't need to extract
                if (!created) return false
            } else {
                val data = ByteArray(bufferSize)
                val fileOutStream = FileOutputStream("$directory/${entry.name}", false)
                val dest = BufferedOutputStream(fileOutStream, bufferSize)
                var count = tarIn.read(data, 0, bufferSize)
                while (count != -1) {
                    dest.write(data, 0, count)
                    try {
                        count = tarIn.read(data, 0, bufferSize)
                    } catch (error: EOFException) { count = -1; stop = 1 }
                }
                dest.close()
            }
            entry = if (stop == 1) {
                null
            } else {
                tarIn.nextTarEntry
            }
        }
        tarIn.close()
        return true
    }
}
