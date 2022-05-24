/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.data.loaders

import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.URL
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream

val DATASET_DIR = System.getProperty("user.home") + "/.diffkt/datasets"

/**
 * Propagates the HTTP_PROXY and HTTPS_PROXY env vars to Java system
 * properties for use by Java network calls.
 */
fun propagateProxy() {
    val httpProxy = System.getenv("http_proxy")
    if (httpProxy != null) {
        val proxyUrl = URL(httpProxy)
        System.setProperty("http.proxyHost", proxyUrl.host)
        System.setProperty("http.proxyPort", proxyUrl.port.toString())
    }
    val httpsProxy = System.getenv("https_proxy")
    if (httpsProxy != null) {
        val proxyUrl = URL(httpsProxy)
        System.setProperty("https.proxyHost", proxyUrl.host)
        System.setProperty("https.proxyPort", proxyUrl.port.toString())
    }
}

/**
 * Downloads a file at the given url to the given directory with the given file name.
 *
 * Uses http/https proxy settings from env vars.
 */
fun downloadFile(url: String, directoryName: String, fileName: String) {
    propagateProxy()
    File(directoryName).mkdirs()
    val f = File("$directoryName/$fileName")
    val input = GZIPInputStream(URL(url).openStream())
    val output = GZIPOutputStream(FileOutputStream(f))
    val dataBuffer = ByteArray(1024)
    var bytesRead = input.read(dataBuffer, 0, 1024)
    while (bytesRead != -1) {
        output.write(dataBuffer, 0, bytesRead)
        bytesRead = input.read(dataBuffer, 0, 1024)
    }
    input.close()
    output.close()
}

/**
 * Downloads a zip file at the given url and extracts it to the given directory
 *
 * uses http/https proxy settings from env vars.
 */
fun downloadZip(url: String, directoryName: String) {
    propagateProxy()
    File(directoryName).mkdirs()
    val outDir = Paths.get(directoryName)
    val input = ZipInputStream(URL(url).openStream())

    var zipEntry = input.nextEntry
    while (zipEntry != null) {
        val newPath = zipSlipValidation(zipEntry, outDir)
        if (zipEntry.isDirectory) {
            Files.createDirectories(newPath)
        } else {
            newPath.parent?.let {
                if (Files.notExists(it))
                    Files.createDirectories(it)
            }
            Files.copy(input, newPath, StandardCopyOption.REPLACE_EXISTING)
        }
        zipEntry = input.nextEntry
    }
}

private fun zipSlipValidation(zipEntry: ZipEntry, outputPath: Path): Path {
    val outputPathResolved = outputPath.resolve(zipEntry.name)
    val normalizedPath = outputPathResolved.normalize()
    if (!normalizedPath.startsWith(outputPath))
        throw IOException("Bad zip entry: ${zipEntry.name}")
    return normalizedPath
}
