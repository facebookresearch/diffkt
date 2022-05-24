/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeExactly
import testutils.*

class DifferentiableTest : AnnotationSpec() {
    @Test fun demoPrimalAndSecondDerivative01() {
        fun f(x: DScalar) = x.pow(3)
        fun fd(x: DScalar) = primalAndForwardDerivative(x, ::f)
        fun fdd(x: DScalar) = primalAndForwardDerivative<DScalar, Pair<DScalar, DScalar>, Pair<DScalar, DScalar>>(
            x = x,
            f = ::fd,
            extractDerivative = { xx: DScalar, d1: Pair<DScalar, DScalar>, extractDerivative: (input: DTensor, output: DTensor) -> DTensor ->
                Pair(extractDerivative(xx, d1.first) as DScalar, extractDerivative(xx, d1.second) as DScalar)
            }
        )
        val r = fdd(FloatScalar(1f))
        val primal = r.first.first
        val firstD = r.first.second // also r.second.first
        firstD shouldBeExactly r.second.first
        val secondD = r.second.second
        primal.value shouldBeExactly 1F
        firstD.value shouldBeExactly 3F
        secondD.value shouldBeExactly 6F
    }
    @Test fun demoPrimalAndSecondDerivative02() {
        fun f(x: DScalar) = x.pow(3)
        fun fd(x: DScalar) = primalAndReverseDerivative(x, ::f)
        fun fdd(x: DScalar) = primalAndReverseDerivative<DScalar, Pair<DScalar, DScalar>, Pair<DScalar, DScalar>>(
            x = x,
            f = ::fd,
            extractDerivative = { xx: DScalar, d1: Pair<DScalar, DScalar>, extractDerivative: (input: DTensor, output: DTensor) -> DTensor ->
                Pair(extractDerivative(xx, d1.first) as DScalar, extractDerivative(xx, d1.second) as DScalar)
            }
        )
        val r = fdd(FloatScalar(1f))
        val primal = r.first.first
        val firstD = r.first.second // also r.second.first
        firstD shouldBeExactly r.second.first
        val secondD = r.second.second
        primal.value shouldBeExactly 1F
        firstD.value shouldBeExactly 3F
        secondD.value shouldBeExactly 6F
    }
    @Test fun reverseDifferentiableByInheritance() {
        class Input(val x: DScalar, val y: DScalar) : Differentiable<Input> {
            override fun wrap(wrapper: Wrapper): Input {
                return Input(wrapper.wrap(x), wrapper.wrap(y))
            }
        }
        class Output(val sum: DScalar, val product: DScalar) : Differentiable<Output> {
            override fun wrap(wrapper: Wrapper): Output {
                return Output(wrapper.wrap(sum), wrapper.wrap(product))
            }
        }
        class Derivative(val dSumDx: DScalar, val dSumDy: DScalar, val dProdDx: DScalar, val dProdDy: DScalar)
        fun f(input: Input): Output {
            return Output(5F * input.x + 7F * input.y, 11F * input.x * input.y)
        }
        fun makeDerivative(input: Input, output: Output, extractDerivative: (DTensor, DTensor) -> DTensor): Derivative {
            return Derivative(
                dSumDx = extractDerivative(input.x, output.sum) as DScalar,
                dSumDy = extractDerivative(input.y, output.sum) as DScalar,
                dProdDx = extractDerivative(input.x, output.product) as DScalar,
                dProdDy = extractDerivative(input.y, output.product) as DScalar,
            )
        }
        val x = Input(FloatScalar(2F), FloatScalar(3F))
        val (p, d) = primalAndReverseDerivative(
            x = x,
            f = ::f,
            extractDerivative = ::makeDerivative)
        p.sum.value shouldBeExactly 31F
        p.product.value shouldBeExactly 66F
        d.dSumDx.value shouldBeExactly 5F
        d.dSumDy.value shouldBeExactly 7F
        d.dProdDx.value shouldBeExactly 33F
        d.dProdDy.value shouldBeExactly 22F
    }

    @Test fun reverseDifferentiableByLambda() {
        class Input(val x: DScalar, val y: DScalar) {
            fun wrap(wrapper: Wrapper): Input {
                return Input(wrapper.wrap(x), wrapper.wrap(y))
            }
        }
        class Output(val sum: DScalar, val product: DScalar) {
            fun wrap(wrapper: Wrapper): Output {
                return Output(wrapper.wrap(sum), wrapper.wrap(product))
            }
        }
        class Derivative(val dSumDx: DScalar, val dSumDy: DScalar, val dProdDx: DScalar, val dProdDy: DScalar)
        fun f(input: Input): Output {
            return Output(5F * input.x + 7F * input.y, 11F * input.x * input.y)
        }
        fun makeDerivative(input: Input, output: Output, extractDerivative: (DTensor, DTensor) -> DTensor): Derivative {
            return Derivative(
                dSumDx = extractDerivative(input.x, output.sum) as DScalar,
                dSumDy = extractDerivative(input.y, output.sum) as DScalar,
                dProdDx = extractDerivative(input.x, output.product) as DScalar,
                dProdDy = extractDerivative(input.y, output.product) as DScalar,
            )
        }
        val x = Input(FloatScalar(2F), FloatScalar(3F))
        val (p, d) = primalAndReverseDerivative(
            x = x,
            f = ::f,
            wrapInput = { i, w -> i.wrap(w) },
            wrapOutput = { o, w -> o.wrap(w) },
            extractDerivative = ::makeDerivative)
        p.sum.value shouldBeExactly 31F
        p.product.value shouldBeExactly 66F
        d.dSumDx.value shouldBeExactly 5F
        d.dSumDy.value shouldBeExactly 7F
        d.dProdDx.value shouldBeExactly 33F
        d.dProdDy.value shouldBeExactly 22F
    }

    @Test fun reverseDifferentiableByRegistration() {
        class Input(val x: DScalar, val y: DScalar) {
            fun wrap(wrapper: Wrapper): Input {
                return Input(wrapper.wrap(x), wrapper.wrap(y))
            }
        }
        class Output(val sum: DScalar, val product: DScalar) {
            fun wrap(wrapper: Wrapper): Output {
                return Output(wrapper.wrap(sum), wrapper.wrap(product))
            }
        }
        Wrapper.register(Input::class, { i, w -> i.wrap(w) })
        Wrapper.register(Output::class, { o, w -> o.wrap(w) })
        class Derivative(val dSumDx: DScalar, val dSumDy: DScalar, val dProdDx: DScalar, val dProdDy: DScalar)
        fun f(input: Input): Output {
            return Output(5F * input.x + 7F * input.y, 11F * input.x * input.y)
        }
        fun makeDerivative(input: Input, output: Output, extractDerivative: (DTensor, DTensor) -> DTensor): Derivative {
            return Derivative(
                dSumDx = extractDerivative(input.x, output.sum) as DScalar,
                dSumDy = extractDerivative(input.y, output.sum) as DScalar,
                dProdDx = extractDerivative(input.x, output.product) as DScalar,
                dProdDy = extractDerivative(input.y, output.product) as DScalar,
            )
        }
        val x = Input(FloatScalar(2F), FloatScalar(3F))
        val (p, d) = primalAndReverseDerivative(
            x = x,
            f = ::f,
            extractDerivative = ::makeDerivative)
        p.sum.value shouldBeExactly 31F
        p.product.value shouldBeExactly 66F
        d.dSumDx.value shouldBeExactly 5F
        d.dSumDy.value shouldBeExactly 7F
        d.dProdDx.value shouldBeExactly 33F
        d.dProdDy.value shouldBeExactly 22F
    }

    @Test fun reverseDifferentiableByPair() {
        fun f(input: Pair<DScalar, DScalar>): Pair<DScalar, DScalar> {
            return Pair(5F * input.first + 7F * input.second, 11F * input.first * input.second)
        }
        fun makeDerivative(input: Pair<DScalar, DScalar>, output: Pair<DScalar, DScalar>, extractDerivative: (DTensor, DTensor) -> DTensor): Pair<Pair<DScalar, DScalar>, Pair<DScalar, DScalar>> {
            return Pair(
                Pair(
                    extractDerivative(input.first, output.first) as DScalar,
                    extractDerivative(input.second, output.first) as DScalar,
                ),
                Pair(
                    extractDerivative(input.first, output.second) as DScalar,
                    extractDerivative(input.second, output.second) as DScalar
                )
            )
        }
        val x = Pair(FloatScalar(2F), FloatScalar(3F))
        val (p, d) = primalAndReverseDerivative(
            x = x,
            f = ::f,
            extractDerivative = ::makeDerivative)
        p.first.value shouldBeExactly 31F
        p.second.value shouldBeExactly 66F
        d.first.first.value shouldBeExactly 5F
        d.first.second.value shouldBeExactly 7F
        d.second.first.value shouldBeExactly 33F
        d.second.second.value shouldBeExactly 22F
    }

    @Test fun reverseDifferentiableByList() {
        fun f(input: List<DScalar>): List<DScalar> {
            return listOf(5F * input[0] + 7F * input[1], 11F * input[0] * input[1])
        }
        fun makeDerivative(input: List<DScalar>, output: List<DScalar>, extractDerivative: (DTensor, DTensor) -> DTensor): List<DScalar> {
            return listOf(
                extractDerivative(input[0], output[0]) as DScalar,
                extractDerivative(input[1], output[0]) as DScalar,
                extractDerivative(input[0], output[1]) as DScalar,
                extractDerivative(input[1], output[1]) as DScalar
            )
        }
        val x = listOf(FloatScalar(2F), FloatScalar(3F))
        val (p, d) = primalAndReverseDerivative(
            x = x,
            f = ::f,
            extractDerivative = ::makeDerivative)
        p[0].value shouldBeExactly 31F
        p[1].value shouldBeExactly 66F
        d[0].value shouldBeExactly 5F
        d[1].value shouldBeExactly 7F
        d[2].value shouldBeExactly 33F
        d[3].value shouldBeExactly 22F
    }

    @Test fun reverseDifferentiableWithTensors() {
        class Input(val x: DTensor, val y: DTensor) : Differentiable<Input> {
            override fun wrap(wrapper: Wrapper): Input {
                return Input(wrapper.wrap(x), wrapper.wrap(y))
            }
        }
        class Output(val sum: DTensor, val product: DTensor) : Differentiable<Output> {
            override fun wrap(wrapper: Wrapper): Output {
                return Output(wrapper.wrap(sum), wrapper.wrap(product))
            }
        }
        class Derivative(val dSumDx: DTensor, val dSumDy: DTensor, val dProdDx: DTensor, val dProdDy: DTensor)
        fun f(input: Input): Output {
            return Output(5F * input.x + 7F * input.y, 11F * input.x * input.y)
        }
        fun makeDerivative(input: Input, output: Output, extractDerivative: (DTensor, DTensor) -> DTensor): Derivative {
            return Derivative(
                dSumDx = extractDerivative(input.x, output.sum),
                dSumDy = extractDerivative(input.y, output.sum),
                dProdDx = extractDerivative(input.x, output.product),
                dProdDy = extractDerivative(input.y, output.product),
            )
        }
        val x = Input(
                tensorOf(1F, 2F, 3F, 4F, 5F, 6F).reshape(2, 3),
                tensorOf(7F, 8F, 9F, 10F, 11F, 12F).reshape(2, 3)
        )
        val (p, d) = primalAndReverseDerivative(
            x = x,
            f = ::f,
            extractDerivative = ::makeDerivative)
        p.sum shouldBeExactly tensorOf(54.0F, 66.0F, 78.0F, 90.0F, 102.0F, 114.0F).reshape(2, 3)
        p.product shouldBeExactly tensorOf(77.0F, 176.0F, 297.0F, 440.0F, 605.0F, 792.0F).reshape(2, 3)

        d.dSumDx shouldBeExactly identityGradientofSameKind(FloatScalar.ONE, Shape(2, 3)) * 5F
        d.dSumDy shouldBeExactly identityGradientofSameKind(FloatScalar.ONE, Shape(2, 3)) * 7F
        d.dProdDx shouldBeExactly tensorOf(
                77.0F, 0.0F,
                0.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 88.0F,
                0.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                99.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 110.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 0.0F,
                121.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 0.0F,
                0.0F, 132.0F).reshape(2, 3, 2, 3)
        d.dProdDy shouldBeExactly tensorOf(
                11.0F, 0.0F,
                0.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 22.0F,
                0.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                33.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 44.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 0.0F,
                55.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 0.0F,
                0.0F, 66.0F).reshape(2, 3, 2, 3)
    }

    @Test fun forwardDifferentiableByInheritance() {
        class Input(val x: DScalar, val y: DScalar) : Differentiable<Input> {
            override fun wrap(wrapper: Wrapper): Input {
                return Input(wrapper.wrap(x), wrapper.wrap(y))
            }
        }
        class Output(val sum: DScalar, val product: DScalar) : Differentiable<Output> {
            override fun wrap(wrapper: Wrapper): Output {
                return Output(wrapper.wrap(sum), wrapper.wrap(product))
            }
        }
        class Derivative(val dSumDx: DScalar, val dSumDy: DScalar, val dProdDx: DScalar, val dProdDy: DScalar)
        fun f(input: Input): Output {
            return Output(5F * input.x + 7F * input.y, 11F * input.x * input.y)
        }
        fun makeDerivative(input: Input, output: Output, extractDerivative: (DTensor, DTensor) -> DTensor): Derivative {
            return Derivative(
                dSumDx = extractDerivative(input.x, output.sum) as DScalar,
                dSumDy = extractDerivative(input.y, output.sum) as DScalar,
                dProdDx = extractDerivative(input.x, output.product) as DScalar,
                dProdDy = extractDerivative(input.y, output.product) as DScalar,
            )
        }
        val x = Input(FloatScalar(2F), FloatScalar(3F))
        val (p, d) = primalAndForwardDerivative(
            x = x,
            f = ::f,
            extractDerivative = ::makeDerivative)
        p.sum.value shouldBeExactly 31F
        p.product.value shouldBeExactly 66F
        d.dSumDx.value shouldBeExactly 5F
        d.dSumDy.value shouldBeExactly 7F
        d.dProdDx.value shouldBeExactly 33F
        d.dProdDy.value shouldBeExactly 22F
    }

    @Test fun forwardDifferentiableByLambda() {
        class Input(val x: DScalar, val y: DScalar) {
            fun wrap(wrapper: Wrapper): Input {
                return Input(wrapper.wrap(x), wrapper.wrap(y))
            }
        }
        class Output(val sum: DScalar, val product: DScalar) {
            fun wrap(wrapper: Wrapper): Output {
                return Output(wrapper.wrap(sum), wrapper.wrap(product))
            }
        }
        class Derivative(val dSumDx: DScalar, val dSumDy: DScalar, val dProdDx: DScalar, val dProdDy: DScalar)
        fun f(input: Input): Output {
            return Output(5F * input.x + 7F * input.y, 11F * input.x * input.y)
        }
        fun makeDerivative(input: Input, output: Output, extractDerivative: (DTensor, DTensor) -> DTensor): Derivative {
            return Derivative(
                dSumDx = extractDerivative(input.x, output.sum) as DScalar,
                dSumDy = extractDerivative(input.y, output.sum) as DScalar,
                dProdDx = extractDerivative(input.x, output.product) as DScalar,
                dProdDy = extractDerivative(input.y, output.product) as DScalar,
            )
        }
        val x = Input(FloatScalar(2F), FloatScalar(3F))
        val (p, d) = primalAndForwardDerivative(
            x = x,
            f = ::f,
            wrapInput = { i, w -> i.wrap(w) },
            wrapOutput = { o, w -> o.wrap(w) },
            extractDerivative = ::makeDerivative)
        p.sum.value shouldBeExactly 31F
        p.product.value shouldBeExactly 66F
        d.dSumDx.value shouldBeExactly 5F
        d.dSumDy.value shouldBeExactly 7F
        d.dProdDx.value shouldBeExactly 33F
        d.dProdDy.value shouldBeExactly 22F
    }

    @Test fun forwardDifferentiableByRegistration() {
        class Input(val x: DScalar, val y: DScalar) {
            fun wrap(wrapper: Wrapper): Input {
                return Input(wrapper.wrap(x), wrapper.wrap(y))
            }
        }
        class Output(val sum: DScalar, val product: DScalar) {
            fun wrap(wrapper: Wrapper): Output {
                return Output(wrapper.wrap(sum), wrapper.wrap(product))
            }
        }
        Wrapper.register(Input::class, { i, w -> i.wrap(w) })
        Wrapper.register(Output::class, { o, w -> o.wrap(w) })
        class Derivative(val dSumDx: DScalar, val dSumDy: DScalar, val dProdDx: DScalar, val dProdDy: DScalar)
        fun f(input: Input): Output {
            return Output(5F * input.x + 7F * input.y, 11F * input.x * input.y)
        }
        fun makeDerivative(input: Input, output: Output, extractDerivative: (DTensor, DTensor) -> DTensor): Derivative {
            return Derivative(
                dSumDx = extractDerivative(input.x, output.sum) as DScalar,
                dSumDy = extractDerivative(input.y, output.sum) as DScalar,
                dProdDx = extractDerivative(input.x, output.product) as DScalar,
                dProdDy = extractDerivative(input.y, output.product) as DScalar,
            )
        }
        val x = Input(FloatScalar(2F), FloatScalar(3F))
        val (p, d) = primalAndForwardDerivative(
            x = x,
            f = ::f,
            extractDerivative = ::makeDerivative)
        p.sum.value shouldBeExactly 31F
        p.product.value shouldBeExactly 66F
        d.dSumDx.value shouldBeExactly 5F
        d.dSumDy.value shouldBeExactly 7F
        d.dProdDx.value shouldBeExactly 33F
        d.dProdDy.value shouldBeExactly 22F
    }

    @Test fun forwardDifferentiableByPair() {
        fun f(input: Pair<DScalar, DScalar>): Pair<DScalar, DScalar> {
            return Pair(5F * input.first + 7F * input.second, 11F * input.first * input.second)
        }
        fun makeDerivative(input: Pair<DScalar, DScalar>, output: Pair<DScalar, DScalar>, extractDerivative: (DTensor, DTensor) -> DTensor): Pair<Pair<DScalar, DScalar>, Pair<DScalar, DScalar>> {
            return Pair(
                Pair(
                    extractDerivative(input.first, output.first) as DScalar,
                    extractDerivative(input.second, output.first) as DScalar,
                ),
                Pair(
                    extractDerivative(input.first, output.second) as DScalar,
                    extractDerivative(input.second, output.second) as DScalar
                )
            )
        }
        val x = Pair(FloatScalar(2F), FloatScalar(3F))
        val (p, d) = primalAndForwardDerivative(
            x = x,
            f = ::f,
            extractDerivative = ::makeDerivative)
        p.first.value shouldBeExactly 31F
        p.second.value shouldBeExactly 66F
        d.first.first.value shouldBeExactly 5F
        d.first.second.value shouldBeExactly 7F
        d.second.first.value shouldBeExactly 33F
        d.second.second.value shouldBeExactly 22F
    }

    @Test fun forwardDifferentiableByList() {
        fun f(input: List<DScalar>): List<DScalar> {
            return listOf(5F * input[0] + 7F * input[1], 11F * input[0] * input[1])
        }
        fun makeDerivative(input: List<DScalar>, output: List<DScalar>, extractDerivative: (DTensor, DTensor) -> DTensor): List<DScalar> {
            return listOf(
                extractDerivative(input[0], output[0]) as DScalar,
                extractDerivative(input[1], output[0]) as DScalar,
                extractDerivative(input[0], output[1]) as DScalar,
                extractDerivative(input[1], output[1]) as DScalar
            )
        }
        val x = listOf(FloatScalar(2F), FloatScalar(3F))
        val (p, d) = primalAndForwardDerivative(
            x = x,
            f = ::f,
            extractDerivative = ::makeDerivative)
        p[0].value shouldBeExactly 31F
        p[1].value shouldBeExactly 66F
        d[0].value shouldBeExactly 5F
        d[1].value shouldBeExactly 7F
        d[2].value shouldBeExactly 33F
        d[3].value shouldBeExactly 22F
    }

    @Test fun forwardDifferentiableWithTensors() {
        class Input(val x: DTensor, val y: DTensor) : Differentiable<Input> {
            override fun wrap(wrapper: Wrapper): Input {
                return Input(wrapper.wrap(x), wrapper.wrap(y))
            }
        }
        class Output(val sum: DTensor, val product: DTensor) : Differentiable<Output> {
            override fun wrap(wrapper: Wrapper): Output {
                return Output(wrapper.wrap(sum), wrapper.wrap(product))
            }
        }
        class Derivative(val dSumDx: DTensor, val dSumDy: DTensor, val dProdDx: DTensor, val dProdDy: DTensor)
        fun f(input: Input): Output {
            return Output(5F * input.x + 7F * input.y, 11F * input.x * input.y)
        }
        fun makeDerivative(input: Input, output: Output, extractDerivative: (DTensor, DTensor) -> DTensor): Derivative {
            return Derivative(
                dSumDx = extractDerivative(input.x, output.sum),
                dSumDy = extractDerivative(input.y, output.sum),
                dProdDx = extractDerivative(input.x, output.product),
                dProdDy = extractDerivative(input.y, output.product),
            )
        }
        val x = Input(
                tensorOf(1F, 2F, 3F, 4F, 5F, 6F).reshape(2, 3),
                tensorOf(7F, 8F, 9F, 10F, 11F, 12F).reshape(2, 3)
        )
        val (p, d) = primalAndForwardDerivative(
            x = x,
            f = ::f,
            extractDerivative = ::makeDerivative)
        p.sum shouldBeExactly tensorOf(54.0F, 66.0F, 78.0F, 90.0F, 102.0F, 114.0F).reshape(2, 3)
        p.product shouldBeExactly tensorOf(77.0F, 176.0F, 297.0F, 440.0F, 605.0F, 792.0F).reshape(2, 3)

        d.dSumDx shouldBeExactly identityGradientofSameKind(FloatScalar.ONE, Shape(2, 3)) * 5F
        d.dSumDy shouldBeExactly identityGradientofSameKind(FloatScalar.ONE, Shape(2, 3)) * 7F
        d.dProdDx shouldBeExactly tensorOf(
                77.0F, 0.0F,
                0.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 88.0F,
                0.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                99.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 110.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 0.0F,
                121.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 0.0F,
                0.0F, 132.0F).reshape(2, 3, 2, 3)
        d.dProdDy shouldBeExactly tensorOf(
                11.0F, 0.0F,
                0.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 22.0F,
                0.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                33.0F, 0.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 44.0F,
                0.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 0.0F,
                55.0F, 0.0F,

                0.0F, 0.0F,
                0.0F, 0.0F,
                0.0F, 66.0F).reshape(2, 3, 2, 3)
    }

    @Test fun testReverseDifferentiableObjectWithSamePrimal() {
        data class A(val x: DScalar, val y: DScalar) : Differentiable<A> {
            override fun wrap(wrapper: Wrapper): A {
                return A(wrapper.wrap(x), wrapper.wrap(y))
            }

            fun extractDerivative(output: DScalar, extractTensorDerivative: (DTensor, DTensor) -> DTensor): A {
                return A(extractTensorDerivative(x, output) as DScalar, extractTensorDerivative(y, output) as DScalar)
            }
        }

        val a = A(FloatScalar(1f), FloatScalar(1f))
        fun f(a: A): DScalar = a.x + 2 * a.y

        val (primal, grad) = primalAndReverseDerivative(
            a, ::f,
            extractDerivative = { input, output, extractTensorDerivative -> input.extractDerivative(output, extractTensorDerivative) }
        )

        primal.value shouldBeExactly 3f
        grad.x.value shouldBeExactly 1f
        grad.y.value shouldBeExactly 2f
    }
    @Test fun testReverseDifferentiableObjectWithUnusedInputs() {
        data class A(val x: DScalar, val y: DScalar) : Differentiable<A> {
            override fun wrap(wrapper: Wrapper): A {
                return A(wrapper.wrap(x), wrapper.wrap(y))
            }

            fun extractDerivative(output: DScalar, extractTensorDerivative: (DTensor, DTensor) -> DTensor): A {
                return A(extractTensorDerivative(x, output) as DScalar, extractTensorDerivative(y, output) as DScalar)
            }
        }

        val a = A(FloatScalar(1f), FloatScalar(1f))

        fun f(a: A): DScalar = a.x
        val (_, grad1) = primalAndReverseDerivative(
            a, ::f,
            extractDerivative = { input, output, extractTensorDerivative -> input.extractDerivative(output, extractTensorDerivative) }
        )
        grad1.x.value shouldBeExactly 1f
        grad1.y.value shouldBeExactly 0f

        @Suppress("UNUSED_PARAMETER")
        fun g(a: A): DScalar = FloatScalar(5f)
        val (_, grad2) = primalAndReverseDerivative(
            a, ::g,
            extractDerivative = { input, output, extractTensorDerivative -> input.extractDerivative(output, extractTensorDerivative) }
        )
        grad2.x.value shouldBeExactly 0f
        grad2.y.value shouldBeExactly 0f
    }
    @Test fun testReverseDifferentiableObjectWithNoInputs() {
        data class A(val x: DScalar, val y: DScalar) : Differentiable<A> {
            override fun wrap(wrapper: Wrapper): A {
                // We don't wrap x and y, treating them as constants for the
                // purposes of differentiation.
                return A(x, y)
            }

            fun extractDerivative(output: DScalar, extractTensorDerivative: (DTensor, DTensor) -> DTensor): A {
                return A(extractTensorDerivative(x, output) as DScalar, extractTensorDerivative(y, output) as DScalar)
            }
        }

        val a = A(FloatScalar(1f), FloatScalar(2f))

        fun f(a: A): DScalar = a.x
        val (_, grad1) = primalAndReverseDerivative(
            a, ::f,
            extractDerivative = { input, output, extractTensorDerivative -> input.extractDerivative(output, extractTensorDerivative) }
        )
        grad1.x.value shouldBeExactly 0f
        grad1.y.value shouldBeExactly 0f

        @Suppress("UNUSED_PARAMETER")
        fun g(a: A): DScalar = FloatScalar(5f)
        val (_, grad2) = primalAndReverseDerivative(
            a, ::g,
            extractDerivative = { input, output, extractTensorDerivative -> input.extractDerivative(output, extractTensorDerivative) }
        )
        grad2.x.value shouldBeExactly 0f
        grad2.y.value shouldBeExactly 0f
    }
    @Test fun testForwardDifferentiableObjectWithUnusedInputs() {
        data class A(val x: DScalar, val y: DScalar) : Differentiable<A> {
            override fun wrap(wrapper: Wrapper): A {
                return A(wrapper.wrap(x), wrapper.wrap(y))
            }

            fun extractDerivative(output: DScalar, extractTensorDerivative: (DTensor, DTensor) -> DTensor): A {
                return A(extractTensorDerivative(x, output) as DScalar, extractTensorDerivative(y, output) as DScalar)
            }
        }

        val a = A(FloatScalar(1f), FloatScalar(1f))

        fun f(a: A): DScalar = a.x
        val (_, grad1) = primalAndForwardDerivative(
            a, ::f,
            extractDerivative = { input, output, extractTensorDerivative -> input.extractDerivative(output, extractTensorDerivative) }
        )
        grad1.x.value shouldBeExactly 1f
        grad1.y.value shouldBeExactly 0f

        @Suppress("UNUSED_PARAMETER")
        fun g(a: A): DScalar = FloatScalar(5f)
        val (_, grad2) = primalAndForwardDerivative(
            a, ::g,
            extractDerivative = { input, output, extractTensorDerivative -> input.extractDerivative(output, extractTensorDerivative) }
        )
        grad2.x.value shouldBeExactly 0f
        grad2.y.value shouldBeExactly 0f
    }
    @Test fun testForwardDifferentiableObjectWithNoInputs() {
        data class A(val x: DScalar, val y: DScalar) : Differentiable<A> {
            override fun wrap(wrapper: Wrapper): A {
                // We don't wrap x and y, treating them as constants for the
                // purposes of differentiation.
                return A(x, y)
            }

            fun extractDerivative(output: DScalar, extractTensorDerivative: (DTensor, DTensor) -> DTensor): A {
                return A(extractTensorDerivative(x, output) as DScalar, extractTensorDerivative(y, output) as DScalar)
            }
        }

        val a = A(FloatScalar(1f), FloatScalar(2f))

        fun f(a: A): DScalar = a.x
        val (_, grad1) = primalAndForwardDerivative(
            a, ::f,
            extractDerivative = { input, output, extractTensorDerivative -> input.extractDerivative(output, extractTensorDerivative) }
        )
        grad1.x.value shouldBeExactly 0f
        grad1.y.value shouldBeExactly 0f

        @Suppress("UNUSED_PARAMETER")
        fun g(a: A): DScalar = FloatScalar(5f)
        val (_, grad2) = primalAndForwardDerivative(
            a, ::g,
            extractDerivative = { input, output, extractTensorDerivative -> input.extractDerivative(output, extractTensorDerivative) }
        )
        grad2.x.value shouldBeExactly 0f
        grad2.y.value shouldBeExactly 0f
    }
}
