/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import org.diffkt.*

fun main() {

    data class Point(val x: Float, val y: Float) {
        constructor(x: Int, y: Int): this(x.toFloat(), y.toFloat())
    }

    val trainingData = listOf(
        Point(1,5),
        Point(2,10),
        Point(3,10),
        Point(4,15),
        Point(5,14),
        Point(6,15),
        Point(7,19),
        Point(8,18),
        Point(9,25),
        Point(10,23)
    )

    data class Tangent(val dm: DScalar, val db: DScalar) {
        operator fun times(float: Float) = Tangent(dm * float, db * float)
    }

    // a simple line with slope and intercept
    data class Line(val m: DScalar, val b: DScalar): Differentiable<Line> {
        constructor(m: Float, b: Float): this(FloatScalar(m), FloatScalar(b))
        override fun wrap(wrapper: Wrapper) = Line(wrapper.wrap(m), wrapper.wrap(b))

        operator fun plus(tangent: Tangent) = Line(m + tangent.dm, b + tangent.db)
        operator fun minus(tangent: Tangent) = Line(m - tangent.dm, b - tangent.db)
    }

    // calculates sum of squares for given training x and y values and a line
    fun sumOfSquares(line: Line): DScalar =
        trainingData.map {
                (it.y - (it.x * line.m + line.b)).pow(2f)
            }.reduce { x1,x2 -> x1 + x2 }

    // declare the line
    var line = Line(0f, 0f)

    // The learning rate
    val lr = .0025F

    // The number of iterations to perform gradient descent
    val iterations = 1000

    // Perform gradient descent
    for (i in 0..iterations) {

        val (p, d) = primalAndReverseDerivatives(
            x = line,
            f = ::sumOfSquares,
            extractDerivative= { input, output, extractTangent ->
                Tangent(
                    dm = extractTangent(input.m, output) as DScalar,
                    db = extractTangent(input.b, output) as DScalar
                )
            }
        )
        line -= d * lr
    }

    println(line)
}