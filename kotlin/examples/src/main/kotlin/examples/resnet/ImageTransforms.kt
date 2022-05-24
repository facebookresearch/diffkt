/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.resnet

import org.diffkt.*
import kotlin.math.min
import kotlin.random.Random

private const val widthAxis = 0
private const val heightAxis = 1
private const val channelsAxis = 2

/**
 * Adds padding to the square image and then crops it to a square of the same size
 */
fun randomSquareCropSameSize(image: DTensor, padding: Int = 0, random: Random): DTensor {
    require(image.shape[widthAxis] == image.shape[heightAxis]) { "Did not pass in a square image" }
    return randomSquareCrop(image, image.shape[widthAxis], padding, random)
}

/**
 * Adds padding to the image and then crops it to a square of shape [size x size]
 */
fun randomSquareCrop(image: DTensor, size: Int, padding: Int = 0, random: Random): DTensor {
    require(image.rank == 3) { "Expected image dimensions are (w x h x channels), received ${image.shape}" }
    require(min(image.shape[widthAxis], image.shape[heightAxis]) + padding >= size) {
        "Desired image size greater than input image size + padding" }

    val vertPadShape = Shape(image.shape[widthAxis], padding, image.shape[channelsAxis])
    val horizPadShape = Shape(padding, image.shape[heightAxis] + padding * 2, image.shape[channelsAxis])
    val vertPad = FloatTensor.zeros(vertPadShape)
    val horizPad = FloatTensor.zeros(horizPadShape)
    val verticallyPaddedImage = concat(listOf(vertPad, image, vertPad), heightAxis)
    val paddedImage = concat(listOf(horizPad, verticallyPaddedImage, horizPad), widthAxis)

    // Start horizontally on the image anywhere that provides at least size subsequent pixels
    val randomStartHoriz = random.nextInt(image.shape[widthAxis] + 2 * padding - size)
    // Start vertically on the image anywhere that provides at least size subsequent pixels
    val randomStartVert = random.nextInt(image.shape[heightAxis] + 2 * padding - size)

    var croppedImage = paddedImage.slice(randomStartHoriz, randomStartHoriz + size, widthAxis)
    croppedImage = croppedImage.slice(randomStartVert, randomStartVert + size, heightAxis)
    return croppedImage
}

/** Flips the input along the provided axis with probability prob */
fun randomFlip(image: DTensor, axis: Int, prob: Float = 0.5f, random: Random): DTensor {
    return if (random.nextFloat() < prob) image.flip(intArrayOf(axis)) else image
}

fun squareImageRandomFlip(image: DTensor, axis: Int = heightAxis, prob: Float = 0.5f, random: Random) =
    randomFlip(image, axis, prob, random)
