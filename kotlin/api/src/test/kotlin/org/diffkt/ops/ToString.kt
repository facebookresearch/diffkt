/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.ops

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe
import org.diffkt.*
import testutils.floats

class ToString : AnnotationSpec() {
    @Test fun `test the toString printed form`() {
        FloatScalar.PI.toString() shouldBe "3.1415927"
        FloatScalar.PI.reshape(1).toString() shouldBe "[3.1415927]"
        FloatScalar.PI.reshape(1, 1).toString() shouldBe "[[3.1415927]]"
        tensorOf(*floats(3)).toString() shouldBe "[1.0, 2.0, 3.0]"
        tensorOf(*floats(3)).reshape(3, 1).toString() shouldBe "[[1.0], [2.0], [3.0]]"
        tensorOf(*floats(3)).reshape(1, 3).toString() shouldBe "[[1.0, 2.0, 3.0]]"
        tensorOf(*floats(6)).reshape(2, 3).toString() shouldBe "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]"
        tensorOf(*floats(6)).reshape(3, 2).toString() shouldBe "[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]"
    }
}
fun main() {
    ToString().`test the toString printed form`()
}