/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.poissonBlending

import org.diffkt.*
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

private const val DEFAULT_SCALE = .5

fun loadImage(filePath: String, scale: Double = DEFAULT_SCALE, offsetX: Int = 0, offsetY: Int = 0): BufferedImage {
    val file = File(filePath)
    return ImageIO.read(file).let {
        val newWidth = (it.width * scale).toInt()
        val newHeight = (it.height * scale).toInt()
        val newImage = BufferedImage(newWidth, newHeight, BufferedImage.TYPE_3BYTE_BGR)
        newImage.createGraphics().run {
            drawImage(it, (offsetX * scale).toInt(), (offsetY * scale).toInt(), newWidth, newHeight, null)
            dispose()
        }
        newImage
    }
}

fun BufferedImage.toTensor(): DTensor {
    val w = this.width
    val h = this.height
    val dataArray = FloatArray(w * h * 3)
    for (j in 0 until h) {
        for (i in 0 until w) {
            val (r, g, b) = this.raster.getPixel(i, j, FloatArray(3))
            val startIndex = (i * h + j) * 3
            dataArray[startIndex] = r
            dataArray[startIndex + 1] = g
            dataArray[startIndex + 2] = b
        }
    }
    return tensorOf(*dataArray).reshape(Shape(w, h, 3))
}


fun getResourcePath(fileName: String): String {
    return Thread.currentThread().contextClassLoader.getResource(fileName)?.path ?: throw Exception("Resource file $fileName not found.")
}

private fun DTensor.asFloat() = (this as DScalar).basePrimal().value

private fun Int.bound() = when {
    this > 255 -> 255
    this < 0 -> 0
    else -> this
}

fun saveTensorAsImage(imageTensor: DTensor, filePath: String) {
    val shape = imageTensor.shape
    val w = shape[0]
    val h = shape[1]
    val outputImage = BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR).also {
        for (j in 0 until h) {
            for (i in 0 until w) {
                val rgb = imageTensor[i, j]
                val red = (rgb[0].asFloat()).toInt().bound()
                val green = (rgb[1].asFloat()).toInt().bound()
                val blue = (rgb[2].asFloat()).toInt().bound()
                val color = (red shl 16) or (green shl 8) or blue
                it.setRGB(i, j, color)
            }
        }
    }
    saveImage(outputImage, filePath)
}

fun saveImage(image: BufferedImage, filePath: String) {
    ImageIO.write(image, "png", File(filePath))
    println("Image saved to $filePath")
}

/**
 * Pastes the masked target onto the base image.
 */
fun pasteImage(base: BufferedImage, target: BufferedImage, mask: BufferedImage): BufferedImage {
    val h = base.height
    val w = base.width
    return BufferedImage(w,h,BufferedImage.TYPE_3BYTE_BGR).also {
        for (j in 0 until h) {
            for (i in 0 until w) {
                val color = if(mask.getRGB(i, j) == Color.BLACK.rgb) base.getRGB(i, j) else target.getRGB(i, j)
                it.setRGB(i, j, color)
            }
        }
    }
}
