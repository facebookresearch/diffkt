/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.brachistochrone

import examples.brachistochrone.tensor.computeTimeTaken
import examples.brachistochrone.tensor.initialY
import examples.brachistochrone.tensor.learn
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeGreaterThan
import io.kotest.matchers.floats.shouldBeLessThan
import org.diffkt.*
import testutils.*

class BrachistochroneTest : AnnotationSpec() {
    @Test
    fun brachistochroneTest() {
        computeTimeTaken(initialY).value shouldBeGreaterThan 1.8F

        val iterations = 400
        val learnedY = (0 until iterations).fold(initialY) { y: DTensor, _: Int ->
            learn(y, 0.1F)
        }

        val rollingTime = computeTimeTaken(learnedY).value
        rollingTime shouldBeGreaterThan 1F
        rollingTime shouldBeLessThan 1.4F
    }
}
