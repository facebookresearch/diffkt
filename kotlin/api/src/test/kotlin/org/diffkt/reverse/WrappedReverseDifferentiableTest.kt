/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeExactly
import org.diffkt.*
import testutils.*

class WrappedReverseDifferentiableTest : AnnotationSpec() {
    @Test
    fun testPullbackWithList() {
        fun f(x: List<DScalar>): DScalar = 5f * x[0] + 4f * x[1]

        val x: List<DScalar> = listOf(FloatScalar(3f), FloatScalar(5f))

        val (p, pullback) = primalAndPullback(x, ::f)

        p.value shouldBeExactly 5f * 3f + 4f * 5f

        val dx = pullback(FloatScalar(1f))

        dx[0].value shouldBeExactly 5f
        dx[1].value shouldBeExactly 4f
    }

    @Test
    fun testLeastSquaresVjp() {
        // tests for some common differentiation operations on non-linear least squares loss functions,
        // useful for the Gauss-Newton method
        fun List<DScalar>.square(): List<DScalar> = this.map { xi -> xi * xi }
        operator fun List<DScalar>.times(c: Float): List<DScalar> = this.map { xi -> c * xi }

        val targets: List<DScalar> = listOf(FloatScalar(4f), FloatScalar(3f))
        fun f(x: List<DScalar>): List<DScalar> = listOf(x[0] * x[1] * 3f, x[0] * x[0] * 9f)
        fun residual(x: List<DScalar>): List<DScalar> = f(x).zip(targets) { a, b -> a - b }
        val c = 0.5f
        fun loss(x: List<DScalar>) = c * residual(x).square().reduce { a, b -> a + b }

        // the gradient of the loss can also be computed as 2 * c * J^T * r
        // where r is the residual and J is the Jacobian of the residual,
        // we actually compute J^T * r as a vjp
        fun lossGradUsingResidualVjp(x: List<DScalar>): List<DScalar> = vjp(x, { it }, ::residual) * 2f * c
        fun lossGrad(x: List<DScalar>): List<DScalar> = reverseDerivative(x, ::loss)

        val x = listOf(FloatScalar(13f), FloatScalar(5f))
        val grad1 = lossGradUsingResidualVjp(x)
        val grad2 = lossGrad(x)
        grad1[0].value shouldBeExactly grad2[0].value
        grad1[1].value shouldBeExactly grad2[1].value
    }

    @Test
    fun testVjpWithUserDefinedTypes1() {
        // assuming that primal and tangent types are the same
        data class Vector2(val x: DScalar, val y: DScalar) : Differentiable<Vector2> {
            override fun wrap(wrapper: Wrapper): Vector2 {
                return Vector2(wrapper.wrap(x), wrapper.wrap(y))
            }

            fun makeReverse(makeReverseTensor: (tensor: DTensor) -> DTensor): Vector2 {
                return Vector2(
                    makeReverseTensor(x) as DScalar,
                    makeReverseTensor(y) as DScalar
                )
            }

            fun extractTangent(extractTensorTangent: (tensor: DTensor) -> (DTensor)): Vector2 {
                return Vector2(
                    extractTensorTangent(x) as DScalar,
                    extractTensorTangent(y) as DScalar
                )
            }
        }

        fun identity(v: Vector2): Vector2 = v

        val v = Vector2(FloatScalar(3f), FloatScalar(5f))
        val dv = Vector2(FloatScalar(8f), FloatScalar(11f))

        val (p, d) = primalAndVjp(
            v, dv, ::identity,
            makeReverseInput = { input: Vector2, makeReverseTensor: (DTensor) -> DTensor ->
                input.makeReverse(makeReverseTensor)
            }
        ) { output: Vector2, extractTensorTangent: (DTensor) -> DTensor ->
            output.extractTangent(extractTensorTangent)
        }

        p.x shouldBeExactly FloatScalar(3f)
        p.y shouldBeExactly FloatScalar(5f)

        d.x shouldBeExactly FloatScalar(8f)
        d.y shouldBeExactly FloatScalar(11f)

        // extractOutputTangent and makeReverseInput should be optional,
        // they can be inferred when the primal and tangent types are the same

        val (p1, d1) = primalAndVjp(v, dv, ::identity)

        p1.x shouldBeExactly FloatScalar(3f)
        p1.y shouldBeExactly FloatScalar(5f)

        d1.x shouldBeExactly FloatScalar(8f)
        d1.y shouldBeExactly FloatScalar(11f)

        fun square(v: Vector2): Vector2 = Vector2(v.x * v.x, v.y * v.y)

        val (p2, d2) = primalAndVjp(v, dv, ::square)

        p2.x shouldBeExactly FloatScalar(9f)
        p2.y shouldBeExactly FloatScalar(25f)

        d2.x shouldBeExactly FloatScalar(2f * 3f * 8f)
        d2.y shouldBeExactly FloatScalar(2f * 5f * 11f)
    }

    @Test
    fun testVjpWithUserDefinedTypes2() {
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

        fun makeReverseInput(primal: Vertex, makeReverseTensor: (DTensor) -> DTensor): Vertex {
            return Vertex(
                makeReverseTensor(primal.pos),
                primal.vel,
                primal.mass
            )
        }

        fun extractTangent(
            input: Vertex,
            extractTensorTangent: (tensor: DTensor) -> (DTensor)
        ): VertexTangent {
            val posTangent = extractTensorTangent(input.pos)
            return VertexTangent(
                posTangent[0] as DScalar,
                posTangent[1] as DScalar
            )
        }

        fun setOutputTangent(
            output: Vertex,
            tangent: VertexTangent,
            setTensorTangent: (tensor: DTensor, tangent: DTensor) -> Unit
        ) {
            val posTangent = meld(tangent.posXTangent, tangent.posYTangent)
            setTensorTangent(output.pos, posTangent)
        }

        fun identity(v: Vertex): Vertex = v

        val vertex = Vertex(tensorOf(1f, 3f), tensorOf(2f, 5f), FloatScalar(7f))
        val vertexTangent = VertexTangent(FloatScalar(10f), FloatScalar(11f))
        val (p, d) = primalAndVjp(
            vertex, vertexTangent, ::identity,
            makeReverseInput = ::makeReverseInput,
            extractInputTangent = ::extractTangent,
            setOutputTangent = ::setOutputTangent
        )

        p.pos shouldBeExactly tensorOf(1f, 3f)
        p.vel shouldBeExactly tensorOf(2f, 5f)
        d.posXTangent.value shouldBeExactly 10f
        d.posYTangent.value shouldBeExactly 11f

        fun square(v: Vertex): Vertex = Vertex(v.pos * v.pos, v.vel * v.vel, v.mass * v.mass)

        val (p2, d2) = primalAndVjp(
            vertex, vertexTangent, ::square,
            makeReverseInput = ::makeReverseInput,
            extractInputTangent = ::extractTangent,
            setOutputTangent = ::setOutputTangent
        )

        p2.pos shouldBeExactly tensorOf(1f, 9f)

        d2.posXTangent.value shouldBeExactly 2f * 1f * 10f
        d2.posYTangent.value shouldBeExactly 2f * 3f * 11f
    }

    @Test
    fun testJvpWithInconsistentPrimalAndTangentObject() {
        // if the user does not provide a custom definition for makeReverseInput,
        // we can infer it if
        // Input == InputTangent == X, and X is Wrappable,
        // but the VJP fails if the primal and tangent objects have a different number of leaves

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
            primalAndVjp(x, xTangent, ::f)
        }
    }

    @Test
    fun testGradientSameTangentClass() {
        // only x and y are differentiable
        class A(val x: DScalar, val y: DScalar, val z: DScalar) : Differentiable<A> {
            override fun wrap(wrapper: Wrapper): A {
                return A(wrapper.wrap(x), wrapper.wrap(y), z)
            }
        }

        fun f(a: A) = a.x * a.x + a.y * a.y + a.z * a.z

        val a = A(FloatScalar(3f), FloatScalar(5f), FloatScalar(2f))

        val (p, grad) = primalAndGradient(a, ::f)

        p.value shouldBeExactly 3f * 3f + 5f * 5f + 2f * 2f

        grad.x.value shouldBeExactly 6f
        grad.y.value shouldBeExactly 10f
        // grad.z.value shouldBeExactly 0f
    }

    @Test
    fun testGradientSameTangentClass2() {
        // only x and y are differentiable
        class A(val x: DScalar, val y: DScalar, val z: DScalar) : Differentiable<A> {
            override fun wrap(wrapper: Wrapper): A {
                return A(wrapper.wrap(x), wrapper.wrap(y), z)
            }

            fun extractTangent(extractTensorTangent: (DTensor) -> DTensor): A {
                return A(extractTensorTangent(x) as DScalar, extractTensorTangent(y) as DScalar, FloatScalar(0f))
            }
        }

        fun f(a: A) = a.x * a.x + a.y * a.y + a.z * a.z

        val a = A(FloatScalar(3f), FloatScalar(5f), FloatScalar(2f))

        val (p, grad) = primalAndGradient(
            a, ::f
        ) { input, extractTensorTangent -> input.extractTangent(extractTensorTangent) }

        p.value shouldBeExactly 3f * 3f + 5f * 5f + 2f * 2f

        grad.x.value shouldBeExactly 6f
        grad.y.value shouldBeExactly 10f
        grad.z.value shouldBeExactly 0f
    }

    @Test
    fun testGradientCustomTangentClass() {
        // only x and y are differentiable
        class ATangent(val x: DScalar, val y: DScalar)
        class A(val x: DScalar, val y: DScalar, val z: DScalar) : Wrappable<A> {
            override fun wrap(wrapper: Wrapper): A {
                return A(wrapper.wrap(x), wrapper.wrap(y), z)
            }

            fun extractTangent(extractTensorTangent: (DTensor) -> DTensor): ATangent {
                return ATangent(extractTensorTangent(x) as DScalar, extractTensorTangent(y) as DScalar)
            }
        }

        fun f(a: A) = a.x * a.x + a.y * a.y + a.z * a.z

        val a = A(FloatScalar(3f), FloatScalar(5f), FloatScalar(2f))

        val (p, grad) = primalAndGradient(
            a, ::f
        ) { input, extractTensorTangent -> input.extractTangent(extractTensorTangent) }

        p.value shouldBeExactly 3f * 3f + 5f * 5f + 2f * 2f

        grad.x.value shouldBeExactly 6f
        grad.y.value shouldBeExactly 10f
    }

    @Test
    fun testGradientCustomTangentClass2() {
        // only x and y are differentiable
        class ATangent(val x: DScalar, val y: DScalar)
        class A(val x: DScalar, val y: DScalar, val z: DScalar) {
            fun makeReverse(makeReverseTensor: (DTensor) -> DTensor): A {
                return A(makeReverseTensor(x) as DScalar, makeReverseTensor(y) as DScalar, z)
            }

            fun extractTangent(extractTensorTangent: (DTensor) -> DTensor): ATangent {
                return ATangent(extractTensorTangent(x) as DScalar, extractTensorTangent(y) as DScalar)
            }
        }

        fun f(a: A) = a.x * a.x + a.y * a.y + a.z * a.z

        val a = A(FloatScalar(3f), FloatScalar(5f), FloatScalar(2f))

        val (p, grad) = primalAndGradient(
            a, ::f,
            makeReverseInput = { input, makeReverseTensor -> input.makeReverse(makeReverseTensor) }
        ) { input, extractTensorTangent -> input.extractTangent(extractTensorTangent) }

        p.value shouldBeExactly 3f * 3f + 5f * 5f + 2f * 2f

        grad.x.value shouldBeExactly 6f
        grad.y.value shouldBeExactly 10f
    }

    @Test
    fun testVjpIf() {
        fun f(x: List<DScalar>): List<DScalar> {
            return x.map { xi -> if (xi.value < 0f) FloatScalar(0f) else xi * xi }
        }

        val x: List<DScalar> = listOf(FloatScalar(9f), FloatScalar(-2f))
        val dy: List<DScalar> = listOf(FloatScalar(1f), FloatScalar(3f))
        val (p, vjp) = primalAndVjp(x, dy, ::f)

        p[0].value shouldBeExactly 81f
        p[1].value shouldBeExactly 0f

        vjp[0].value shouldBeExactly 18f
        vjp[1].value shouldBeExactly 0f
    }
}