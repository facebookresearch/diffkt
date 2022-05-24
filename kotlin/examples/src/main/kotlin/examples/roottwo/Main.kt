/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.roottwo

import org.diffkt.*
import kotlin.random.Random

fun predict(x: DScalar): DScalar {
    return x * x
}

fun cost(x: DScalar, y: Float): DScalar {
    val predicted = predict(x)
    return (predicted - y).let { it * it }
}

fun main() {
    var x = FloatScalar(Random.nextFloat()) as DScalar
    val y = 2f
    val learningRate = 0.01f

    for (i in 0..200) {
        val xGrad = reverseDerivative(x) { xx: DScalar -> cost(xx, y) }
        x -= xGrad * learningRate
        if (i % 20 == 0) println(x)
    }
    println("final x = $x")
    println("Final x * x = " + predict(x))
}
