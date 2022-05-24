/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.neohookean.simple.oo.quasistatic

import org.diffkt.*

class Vertex(val x: DScalar, val y: DScalar): Differentiable<Vertex> {
    constructor(x: Float, y: Float): this(FloatScalar(x), FloatScalar(y))

    override fun wrap(wrapper: Wrapper): Vertex {
        return Vertex(wrapper.wrap(x), wrapper.wrap(y))
    }

    // TODO consider implementing a separate gradient descent optimizer class
    fun gradientDescent(dVertex: Vertex, stepSize: Float): Vertex {
        return Vertex(
            this.x - stepSize * dVertex.x,
            this.y - stepSize * dVertex.y
        )
    }
}

fun primalAndReverseDerivative(
    vertices: List<Vertex>,
    f: (List<Vertex>) -> DScalar
): Pair<DScalar, List<Vertex>> {
    return primalAndReverseDerivative(
        vertices,
        f,
        extractDerivative = { vertices, output, extract ->
            vertices.map { vertex ->
                Vertex(
                    extract(vertex.x, output) as DScalar,
                    extract(vertex.y, output) as DScalar
                )
            }
        }
    )
}