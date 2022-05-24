/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.external

import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe

class PredicateTest : AnnotationSpec() {
    @Test fun ifThenElseTest() {
        val p = floatArrayOf(1f, 0f, 0f, 1f, 1f)
        val a1 = floatArrayOf(2f, 0f, -2f, 3f, 4f)
        val a2 = floatArrayOf(9f, 11.2f, -21f, -2f, 2f)
        Predicate.ifThenElse(p, a1, a2, 5) shouldBe floatArrayOf(2f, 11.2f, -21f, 3f, 4f)
    }
}
