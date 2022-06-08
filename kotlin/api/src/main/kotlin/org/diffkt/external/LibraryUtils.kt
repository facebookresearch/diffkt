/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.external

internal fun loadLib(name: String) {
    fun getExtension(name: String): String {

        val os = System.getProperty("os.name")
        val ext = if (os.contains("Linux", ignoreCase = true)) {
            ".so"
        } else if (os.contains("Darwin", ignoreCase = true)) {
            ".dylib"
        } else if (os.contains("Windows", ignoreCase = true)) {
            ".dll"
        } else
            throw Exception("Unsupported os - ${os}")

        return ext
    }

    val ext = getExtension(name)
    val libFileName = "${name}$ext"

    // code inside the try block is adapted from the accepted answer here:
    // https://stackoverflow.com/questions/4691095/java-loading-dlls-by-a-relative-path-and-hide-them-inside-a-jar
    try {
        val input = Thread.currentThread().contextClassLoader.getResourceAsStream(libFileName)
            ?: throw RuntimeException("Unable to find $libFileName in resources")
        val buffer = ByteArray(1024)
        // We copy the lib to a temp file because we were unable to load the lib directly from within a jar when
        // using the packaged library.
        val temp = createTempFile("utils/${name}$ext", "")
        val fos = temp.outputStream()

        var read = input.read(buffer)
        while (read != -1) {
            fos.write(buffer, 0, read)
            read = input.read(buffer)
        }
        fos.close()
        input.close()

        System.load(temp.absolutePath)
    } catch (e: Exception) {
        throw RuntimeException("Unable to load lib $libFileName: ${e.message}")
    }
}
