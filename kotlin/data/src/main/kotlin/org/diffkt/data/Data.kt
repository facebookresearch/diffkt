/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.data

import org.diffkt.*
import kotlin.math.min
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.random.Random

/**
 * Represents a chunk of data comprised of features and their corresponding labels.
 * Data can optionally include a mask value, which indicates where padding occurs
 * on a feature.
 *
 * If data [1, 2, 3] is padded to be [1, 2, 3, 0, 0], the corresponding mask would be [1, 1, 1, 0, 0]
 */
class Data(
    features: FloatTensor,
    labels: FloatTensor,
    featureMasks: FloatTensor? = null,
    labelMasks: FloatTensor? = null) {

    var features = features
        private set
    var featureMasks = featureMasks
        private set
    var labels = labels
        private set
    var labelMasks = labelMasks
        private set

    fun to(device: Device): Data {
        val f2 = features.to(device)
        val l2 = labels.to(device)
        val fm = featureMasks?.to(device)
        val lm = labelMasks?.to(device)
        return Data(f2, l2, fm, lm)
    }

    init {
        require(features.shape.first == labels.shape.first) {
            "The number of examples must match number of labels." +
                    " Got ${features.shape.first} examples and ${labels.shape.first} labels"
        }
    }

    val size = features.shape[0]

    /**
     * Shuffles individual examples within the Data
     */
    fun shuffle(random: Random) {
        val numBatches = features.shape.first
        assert(labels.shape.first == numBatches)
        val permutation = (0 until numBatches).shuffled(random)
        val shuffledFeatures = features.gather(permutation, 0)
        val shuffledLabels = labels.gather(permutation, 0)
        val shuffledFeatureMasks = featureMasks?.gather(permutation, 0)
        val shuffledLabelMasks = labelMasks?.gather(permutation, 0)
        features = shuffledFeatures
        labels = shuffledLabels
        featureMasks = shuffledFeatureMasks
        labelMasks = shuffledLabelMasks
    }

    override fun equals(other: Any?): Boolean {
        if (other !is Data) return false
        return this.features == other.features && this.labels == other.labels
    }

    /**
     * Prints the features and labels of at most @max examples in data. Features and labels are transformed to Strings
     * for printing according to @featuresToStrings and @labelsToStrings
     */
    fun display(max: Int = 10,
                     featuresToStrings: ((FloatTensor) -> List<String>) = ::getDefaultStrings,
                     labelsToStrings: ((FloatTensor) -> List<String>) = ::getDefaultStrings) {
        val end = min(size, max)
        val featuresToDisplay = features.slice(0, end, 0)
        val labelsToDisplay = labels.slice(0, end, 0)

        val featureStrings = featuresToStrings(featuresToDisplay)
        val labelStrings = labelsToStrings(labelsToDisplay)

        for (i in featureStrings.indices) {
            println("$this example $i:")
            println("--- features: ${featureStrings[i]}")
            println("--- label: ${labelStrings[i]}")
        }
    }
}

/**
 * Default function for use with Data.display
 *
 * Slices @tensor along the batch dimension into individual examples (batch size 1),
 * returns the list of the default (defined by Tensor) String representation of each slice
 *
 * Example usage:
 * data.display(labelsToStrings = ::getDefaultStrings)
 */
fun getDefaultStrings(tensor: FloatTensor) : List<String> {
    val batches = tensor.shape.first
    return (0 until batches).map { i ->
        val slice = tensor.slice(i, i + 1, 0)
        "$slice"
    }
}

/**
 * Display function for images, for use with Data.display.
 *
 * Generates temporary image files for each image in @features, and automatically opens them in Preview. Returns
 * a list of Strings stating the filename, of the form "img_0_tmp4718410819101475184.jpg opened in Preview". Removes the
 * temp files when finished.
 *
 * Assumes tensors are in the format NHWC or NHW; supports both greyscale and color images.
 *
 * Example usage:
 * data.display(featuresToStrings = ::displayImages)
 */
fun displayImages(features: FloatTensor) : List<String> {
    // validate image data
    require(features.rank == 4 || features.rank == 3) {
        "Expected tensor of rank 4 [batches, img_height, img_width, channels] or rank 3 " +
                "[batches, img_height, img_width]; got rank ${features.rank}"
    }
    val numImages = features.shape.first
    val height = features.shape[1]
    val width = features.shape[2]
    val numChannels = features.shape[3]
    val greyscale = numChannels == 1

    require(numChannels == 3 || numChannels == 1) {
        "Expected num channels to be 1 (for greyscale images) or 3 (for color images). Got $numChannels"
    }

    // normalizes all color values to the range [0, 1]
    fun normalizeImageFeatures(features: FloatTensor): FloatTensor {
        val minValue = features.min()
        val maxValue = features.max()
        return ((features - minValue) / (maxValue - minValue)) as FloatTensor
    }

    // normalize image data and get the images to display
    val batchedNormalizedImages = normalizeImageFeatures(features)
    // val normalizedImagesList = (0 until numImages).map { i -> batchedNormalizedImages.slice(i, i + 1, 0) as StridedFloatTensor }

    // for each image tensor, write an image file
    val files = (0 until numImages).map { idx: Int ->
        // create an empty image
        val img = batchedNormalizedImages[idx] as FloatTensor
        val bi = BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)

        // set each pixel's color
        for (y in 0 until height) {
            val row = img[y]
            for (x in 0 until width) {
                val pix = row[x]
                fun component(i: Int) = (pix[i] as FloatScalar).value
                val color = if (greyscale) {
                    val p = component(0)
                    Color(p, p, p)
                } else {
                    Color(component(0), component(1), component(2))
                }

                bi.setRGB(x, y, color.rgb)
            }
        }

        // write the image to a file
        val file = File.createTempFile("img_${idx}_tmp", ".jpg")
        ImageIO.write(bi, "jpg", file)

        // add the new image file to the output list
        file
    }

    // open all the image files in Preview, then delete them
    val paths = files.joinToString(" ")
    val commandStr = "open -a Preview.app $paths; sleep 20; rm -f $paths"
    Runtime.getRuntime().exec(arrayOf("/bin/sh", "-c", commandStr))

    return files.map { "${it.name} opened in Preview" }
}
