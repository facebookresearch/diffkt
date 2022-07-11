/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import java.net.URL
import kotlin.random.Random
import org.diffkt.*
import kotlin.random.nextInt

/**
 * Predict a light/dark font based on R,G,B color background
 */
fun main() {

    // Import CSV data with R,G,B input values and a light/dark indicator  (0,1)
    val allDataTensor = URL("https://tinyurl.com/y2qmhfsr")
        .readText().split(Regex("\\r?\\n"))
        .asSequence()
        .drop(1)
        .filter { it.isNotBlank() }
        .flatMap { s ->
            s.split(",").map { it.toFloat() }
        }.toList()
        .toFloatArray()
        .let { values ->
            val n = values.count() / 4
            tensorOf(*values).reshape(n,4)
        }

    // Extract 3 input columns
    val inputTensor = allDataTensor.view(0..2, 1) / 255f

    // Extract 1 output column
    val outputTensor = allDataTensor.view(3, 1)


    var wHiddenTensor: DTensor = FloatTensor.random(Random, Shape(3,3))
    var wOuterTensor: DTensor = FloatTensor.random(Random, Shape(1,3))

    var bHiddenTensor: DTensor = FloatTensor.random(Random, Shape(3,1))
    var bOuterTensor: DTensor = FloatTensor.random(Random, Shape(1,1))

    // forward propagation neural network
    // provide a single input sample for stochastic gradient descent
    fun neuralNetwork(xSample: DTensor,
                      wHidden: DTensor = wHiddenTensor,
                      wOuter: DTensor = wOuterTensor,
                      bHidden: DTensor = bHiddenTensor,
                      bOuter: DTensor = bOuterTensor
    ): DTensor {
        val middleOutput = relu(wHidden.matmul(xSample.transpose()) + bHidden)
        val outerOutput = sigmoid(wOuter.matmul(middleOutput) + bOuter)
        return outerOutput
    }

    // Calculate loss using sum of squares
    fun loss(xSample: DTensor,
             ySample: DTensor,
             wHidden: DTensor = wHiddenTensor,
             wOuter: DTensor = wOuterTensor,
             bHidden: DTensor = bHiddenTensor,
             bOuter: DTensor = bOuterTensor
    ): DScalar {
        return ((ySample - neuralNetwork(xSample, wHidden, wOuter, bHidden, bOuter)).pow(2)).sum()
    }
    // number of elements
    val n = allDataTensor.shape.first

    // The learning rate
    val lr = .001F

    // The number of iterations to perform gradient descent
    val iterations = 100_000

    // Perform stochastic gradient descent
    for (i in 0..iterations) {

        // sample a random row from input and output tensors
        val randomRow = Random.nextInt(0 until n)
        val xSample = inputTensor[randomRow]
        val ySample = outputTensor[randomRow]

        // get gradients
        val wHiddenGradients = reverseDerivative(wHiddenTensor) { t -> loss(xSample, ySample, wHidden = t) }
        val wOutputGradients = reverseDerivative(wOuterTensor) { t -> loss(xSample, ySample, wOuter = t) }
        val bHiddenGradients = reverseDerivative(bHiddenTensor) { t -> loss(xSample, ySample, bHidden = t) }
        val bOutputGradients = reverseDerivative(bOuterTensor) { t -> loss(xSample, ySample, bOuter = t) }

        // update weights and biases by subtracting their (learning rate) * (gradient)
        wHiddenTensor -= wHiddenGradients * lr
        wOuterTensor -= wOutputGradients * lr
        bHiddenTensor -= bHiddenGradients * lr
        bOuterTensor -= bOutputGradients * lr
    }

    // calculate accuracy
    val accuracy = (neuralNetwork(inputTensor).flatten().gt(0.5F)
        .eq(outputTensor.flatten())).sum() / n.toFloat()

    println(accuracy)
}