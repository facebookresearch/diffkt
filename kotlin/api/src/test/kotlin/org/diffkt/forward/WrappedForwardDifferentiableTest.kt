/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.forward

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeExactly
import org.diffkt.*
import testutils.*
import java.lang.IllegalArgumentException

class WrappedForwardDifferentiableTest : AnnotationSpec() {

    @Test
    fun testJvpWithList() {
        fun square(x: List<DScalar>): List<DScalar> = x.map { xi -> xi * xi }

        val x: List<DScalar> = listOf(FloatScalar(3f), FloatScalar(5f))
        val dx: List<DScalar> = listOf(FloatScalar(9f), FloatScalar(13f))

        fun extractTangent(
            output: List<DScalar>,
            extractTensorTangent: (outputTensor: DTensor) -> (DTensor)
        ): List<DScalar> {
            return output.map { xi -> extractTensorTangent(xi) as DScalar }
        }

        val (p, d) = primalAndJvp(
            x, dx, ::square,
            extractOutputTangent = ::extractTangent
        )

        p[0].value shouldBeExactly 9f
        p[1].value shouldBeExactly 25f

        // jvp of x^2 is { x, dx -> 2 * x * dx }
        d[0].value shouldBeExactly 2f * 3f * 9f
        d[1].value shouldBeExactly 2f * 5f * 13f
    }

    @Test
    fun testJvpWithUserDefinedTypes1() {
        // assuming that primal and tangent types are the same
        data class Vector2(val x: DScalar, val y: DScalar) : Differentiable<Vector2> {
            override fun wrap(wrapper: Wrapper): Vector2 {
                return Vector2(wrapper.wrap(x), wrapper.wrap(y))
            }

            fun extractTangent(extractTensorTangent: (outputTensor: DTensor) -> (DTensor)): Vector2 {
                return Vector2(
                    extractTensorTangent(x) as DScalar,
                    extractTensorTangent(y) as DScalar
                )
            }
        }

        fun identity(v: Vector2): Vector2 = v

        val v = Vector2(FloatScalar(3f), FloatScalar(5f))
        val dv = Vector2(FloatScalar(8f), FloatScalar(11f))

        val (p, d) = primalAndJvp(
            v, dv, ::identity
        ) { output: Vector2, extractTensorTangent: (DTensor) -> DTensor ->
            output.extractTangent(extractTensorTangent)
        }

        p.x shouldBeExactly FloatScalar(3f)
        p.y shouldBeExactly FloatScalar(5f)

        d.x shouldBeExactly FloatScalar(8f)
        d.y shouldBeExactly FloatScalar(11f)

        // extractOutputTangent should be optional, it can be inferred when the primal and tangent types are the same
        val (p1, d1) = primalAndJvp(v, dv, ::identity)

        p1.x shouldBeExactly FloatScalar(3f)
        p1.y shouldBeExactly FloatScalar(5f)

        d1.x shouldBeExactly FloatScalar(8f)
        d1.y shouldBeExactly FloatScalar(11f)

        fun square(v: Vector2): Vector2 = Vector2(v.x * v.x, v.y * v.y)

        val (p2, d2) = primalAndJvp(
            v, dv, ::square
        ) { output: Vector2, extractTensorTangent: (DTensor) -> DTensor ->
            output.extractTangent(extractTensorTangent)
        }

        p2.x shouldBeExactly FloatScalar(9f)
        p2.y shouldBeExactly FloatScalar(25f)

        // jvp of v^2 is { v, dv -> 2 * v * dv }
        d2.x shouldBeExactly FloatScalar(2f * 3f * 8f)
        d2.y shouldBeExactly FloatScalar(2f * 5f * 11f)
    }

    @Test
    fun testJvpWithUserDefinedTypes2() {
        // custom Vertex class attributes:
        // pos (a Vector2 implemented as a tensor) is differentiable
        // vel (a Vector2 implemented as a tensor) is non-differentiable
        // mass is non-differentiable
        data class Vertex(val pos: DTensor, val vel: DTensor, val mass: DScalar)
        // for the tangent object, instead of storing
        // (val posTangent: DTensor, val velTangent: DTensor, val massTangent: DTensor)
        // where velTangent and massTangent are always zero tensors,
        // we store the scalar components of posTangent:
        // (val posXTangent: DScalar, val posYTangent: DScalar)
        // to demonstrate that we can handle arbitrary types, even when there is no one-to-one mapping
        // between tensor leaves in the primal object and tensor leaves in the tangent object
        data class VertexTangent(val posXTangent: DScalar, val posYTangent: DScalar)

        fun makeForwardInput(primal: Vertex, tangent: VertexTangent, makeForwardTensor: (DTensor, DTensor) -> DTensor): Vertex {
            val posTangent = meld(tangent.posXTangent, tangent.posYTangent)
            return Vertex(
                makeForwardTensor(primal.pos, posTangent),
                primal.vel,
                primal.mass
            )
        }

        fun extractTangent(
            output: Vertex,
            extractTensorTangent: (outputTensor: DTensor) -> (DTensor)
        ): VertexTangent {
            val posTangent = extractTensorTangent(output.pos)
            return VertexTangent(
                posTangent[0] as DScalar,
                posTangent[1] as DScalar
            )
        }

        fun identity(v: Vertex): Vertex = v

        val vertex = Vertex(tensorOf(1f, 3f), tensorOf(2f, 5f), FloatScalar(7f))
        val vertexTangent = VertexTangent(FloatScalar(10f), FloatScalar(11f))
        val (p, d) = primalAndJvp(
            vertex, vertexTangent, ::identity,
            makeForwardInput = ::makeForwardInput,
            extractTangent = ::extractTangent
        )

        p.pos shouldBeExactly tensorOf(1f, 3f)
        p.vel shouldBeExactly tensorOf(2f, 5f)
        d.posXTangent.value shouldBeExactly 10f
        d.posYTangent.value shouldBeExactly 11f

        fun square(v: Vertex): Vertex = Vertex(v.pos * v.pos, v.vel * v.vel, v.mass * v.mass)

        val (p2, d2) = primalAndJvp(
            vertex, vertexTangent, ::square,
            makeForwardInput = ::makeForwardInput,
            extractTangent = ::extractTangent
        )

        p2.pos shouldBeExactly tensorOf(1f, 9f)

        // jvp of v^2 is { v, dv -> 2 * v * dv }
        d2.posXTangent.value shouldBeExactly 2f * 1f * 10f
        d2.posYTangent.value shouldBeExactly 2f * 3f * 11f
    }

    @Test
    fun testJvpWithInconsistentPrimalAndTangentObject() {
        // if the user does not provide a custom definition for makeForwardInput,
        // we can infer it if
        // Input == InputTangent == X, and X is Wrappable,
        // but the JVP fails if the primal and tangent objects have a different number of leaves

        class X(val xs: List<DScalar>) : Wrappable<X> {
            override fun wrap(wrapper: Wrapper): X {
                return X(xs.map { xi -> wrapper.wrap(xi) })
            }
        }

        fun f(x: X): X = X(x.xs.map { xi -> xi * 2f })

        val x = X(listOf(FloatScalar(1f)))
        val xTangent = X(listOf(FloatScalar(2f), FloatScalar(4f)))

        // x and xTangent are instances of the same class,
        // but the inferred makeForwardInput fails
        // due to inconsistent number of leaves
        shouldThrow<IllegalArgumentException> {
            primalAndJvp(x, xTangent, ::f)
        }
    }
}