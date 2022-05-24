/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe
import org.diffkt.*
import kotlin.random.Random

class EmbeddingBagTest : AnnotationSpec() {
    @Test fun embeddingBag() {
        val eb = EmbeddingBag(4, 1, EmbeddingBag.Companion.Reduction.Sum, Random(123))

        val (res, embeddingBagTangent) = primalAndReverseDerivative(
            x = eb,
            f = { eb_: EmbeddingBag -> eb_(intTensorOf(0, 1, 3), intTensorOf(0, 2)) },
            extractDerivative = { input: EmbeddingBag,
                                  output: DTensor,
                                  extractor: (tensorInput: DTensor, tensorOutput: DTensor)->DTensor
                ->
                input.trainableWeights.extractTangent(output, extractor)
            }
        )
        val embeddingBagVJP = embeddingBagTangent.value.sum(axes = intArrayOf(2, 3))

        val expected0 = eb(intScalarOf(0), intScalarOf(0)) + eb(intScalarOf(1), intScalarOf(0))
        val expected1 = eb(intScalarOf(3), intScalarOf(0))
        res[0] shouldBe expected0
        res[1] shouldBe expected1

        embeddingBagVJP shouldBe FloatTensor(Shape(4, 1), floatArrayOf(1f, 1f, 0f, 1f))
    }

    @Test fun emptyEmbeddingBag() {
        // TODO: empty embedding bags should result in a vector of zeros
    }

    @Test fun offsetsDontStartWithZero() {
        // TODO: offsets don't start with zero
    }
}
