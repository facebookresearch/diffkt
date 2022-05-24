/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.poissonBlending

import examples.utils.timing.warmupAndTime
import org.diffkt.*
import java.io.File

// Before open sourcing we should make sure that these images can be included. They are originally from the
// Opt-lang repository: https://github.com/niessner/Opt/tree/master/examples/data

val baseImagePath = getResourcePath("poisson0.png")
val resourcesPath: String = File(baseImagePath).parent
val baseImage = loadImage(baseImagePath)
val base = baseImage.toTensor()

val targetImage = loadImage(getResourcePath("poisson1.png"))
val target = targetImage.toTensor()

val maskImage = loadImage(getResourcePath("poisson_mask.png"))
val mask = maskImage.toTensor() / 255f // 1f for white, 0f for black

fun DTensor.shiftLeft(): DTensor {
    require(this.rank == 3)
    val width = this.shape[0]
    val height = this.shape[1]
    val channels = this.shape[2]

    return FloatTensor.zeros(Shape(1, height, channels)).concat(this.slice(0, width - 1, axis = 0), axis = 0)
}

fun DTensor.shiftRight(): DTensor {
    require(this.rank == 3)
    val width = this.shape[0]
    val height = this.shape[1]
    val channels = this.shape[2]
    return this.slice(1, width, axis = 0).concat(FloatTensor.zeros(Shape(1,height,channels)), axis = 0)
}

fun DTensor.shiftDown(): DTensor {
    require(this.rank == 3)
    val width = this.shape[0]
    val height = this.shape[1]
    val channels = this.shape[2]
    return this.slice(1, height, axis = 1).concat(FloatTensor.zeros(Shape(width, 1, channels)), axis = 1)
}

fun DTensor.shiftUp(): DTensor {
    require(this.rank == 3)
    val width = this.shape[0]
    val height = this.shape[1]
    val channels = this.shape[2]
    return FloatTensor.zeros(Shape(width, 1, channels)).concat(this.slice(0, height - 1, axis = 1), axis = 1)
}

fun imageGradient(x: DTensor): DTensor {
    return 4f * x - x.shiftLeft() - x.shiftRight() - x.shiftDown() - x.shiftUp()
}

fun residuals(x: DTensor): DTensor {
    return (imageGradient(x) - imageGradient(target)) * mask
}

fun loss(x: DTensor): DScalar {
    return 0.5f * residuals(x).pow(2).sum()
}

fun run(iters: Int, lr: Float): DTensor {
    var x = base
    repeat(iters) { iter ->
        val grad = reverseDerivative(x, ::loss)
        x -= grad * lr
    }
    return x
}

/**
 * Uses gradient descent.
 */
fun main(args: Array<String>) {
    val iters = args.getOrNull(0)?.toIntOrNull() ?: 5000
    val lr = args.getOrNull(1)?.toFloatOrNull() ?: 0.015f

    val isBenchmark = args.lastOrNull() == "benchmark"
    if (isBenchmark) {
        val time = warmupAndTime({ run(iters, lr) })
        println("===========================================")
        println("POISSON IMAGE EDITING: GRADIENT DESCENT")
        println("iters: $iters  lr: $lr")
        println("time: $time ns")
        println("loss: ${loss(run(iters, lr))}")
        return
    }

    val x = run(iters, lr)
    val outputFilePath = "$resourcesPath/result.png"
    saveTensorAsImage(x, outputFilePath)
}
