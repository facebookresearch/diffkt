/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import java.net.URL
import org.diffkt.*
import kotlin.random.Random

fun main() {

    data class Point(val x1: Float, val x2: Float, val y: Float)

    val points = URL("https://bit.ly/35ebET5")
        .readText().split(Regex("\\r?\\n"))
        .asSequence()
        .drop(1)
        .filter { it.isNotEmpty() }
        .map { it.split(",").map{ it.toFloat()} }
        .map { (x1,x2,y) -> Point(x1,x2,y) }
        .toList()

    // helper function that idiomatically maps a List<T> to a tensor
    fun <T> List<T>.toTensor(vararg mappers: (item: T) -> Float): FloatTensor {

        val rows = count()
        val cols = mappers.count()

        return FloatTensor.invoke(Shape(rows, cols)) { i: Int ->
            mappers[i % cols](this[i / cols])
        }
    }

    // map input variables to tensors,
    // add a placeholder "1" column to generate intercept
    val x = points.toTensor({it.x1}, {it.x2}, {1f})
    val y = points.toTensor({it.y})

    // initialize coefficients
    var betas: DTensor = FloatTensor.random(Random,Shape(1,3))

    // calculate sum of squares of the error with given slope and intercept for a line
    fun loss(betas: DTensor): DScalar =
        (y - x * betas).pow(2).sum()

    // The learning rate
    val lr = .001F

    // The number of iterations to perform gradient descent
    val iterations = 100_000

    // Perform gradient descent
    for (i in 0..iterations) {

        // get gradients for line slope and intercept
        val betaGradients = reverseDerivative(betas, ::loss)

        // update m and b by subtracting the (learning rate) * (slope)
        betas -= betaGradients * lr
    }
    print("betas=$betas")
}
