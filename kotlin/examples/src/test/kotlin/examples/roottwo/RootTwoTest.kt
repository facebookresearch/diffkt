/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.roottwo

import org.diffkt.*
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.floats.shouldBeExactly
import testutils.*

class RootTwoTest : AnnotationSpec() {
    @Test
    fun forwardAndBackward() {
        val x = FloatScalar(0.5f)
        val y = 2f
        val forwardGrad = forwardDerivative(x) { xx: DScalar -> cost(xx, y) }
        val reverseGrad = reverseDerivative(x) { xx: DScalar -> cost(xx, y) }
        forwardGrad.value shouldBeExactly -3.5f
        reverseGrad.value shouldBeExactly -3.5f
    }
}
