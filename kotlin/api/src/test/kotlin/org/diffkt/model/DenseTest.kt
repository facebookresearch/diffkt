/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldNotBe
import io.kotest.matchers.string.shouldContain
import org.diffkt.*
import testutils.*
import kotlin.random.Random

class DenseTest : AnnotationSpec() {
    @Test
    fun passthrough() {
        val dense = Dense(2, 2, Random(0),
            biasInit = { shape: Shape, _: Random ->
                FloatTensor.zeros(shape)
            },
            weightInit = { _: Shape, _: Random ->
                identityGradientofSameKind(FloatScalar.ONE, Shape(2)) as FloatTensor
            }
        )
        val input = FloatTensor(Shape(4, 2), floats(4 * 2))
        dense(input) shouldBeExactly input
    }

    @Test
    fun badInputShape() {
        val dense = Dense(2, 2, Random(0))
        val input = FloatTensor(Shape(2, 3), floats(6))
        shouldThrow<IllegalArgumentException> { dense(input) }
    }

    @Test
    fun badInputRank() {
        val dense = Dense(2, 2, Random(0))
        val input = FloatTensor(Shape(2), floats(2))
        val e = shouldThrow<IllegalArgumentException> { dense(input) }
        e.message shouldContain "rank"
    }

    @Test
    fun noBias() {
        val dense = Dense(2, 2, Random(0), false)
        val weightBefore = dense.w
        val biasBefore = dense.b
        val input = FloatTensor(Shape(2, 2), floats(4))
        val optimizer = SGDOptimizer<DummyModel>(0.1F, weightDecay = true, momentum = true)

        val (_, grads) = primalAndReverseDerivative(
            x = dense,
            // We sum here so f's output is a scalar so our gradients are the
            // right shapes for the adjust().
            f = { dense_ -> dense_(input).sum() },
            extractDerivative = { input2: Dense,
                                  output: DScalar,
                                  extractor: (input: DTensor, output: DTensor)->DTensor ->
                input2.extractTangent(output, extractor)
            }
        )
        val newDense = dense.trainingStep(optimizer, grads)

        val weightAfter = newDense.w
        val biasAfter = newDense.b
        // Confirm that the weight parameter changed with learning
        weightBefore shouldNotBe weightAfter
        // Confirm bias did not change and is still 0
        biasBefore shouldBeExactly biasAfter
        biasAfter shouldBeExactly FloatScalar.ZERO
    }
}
