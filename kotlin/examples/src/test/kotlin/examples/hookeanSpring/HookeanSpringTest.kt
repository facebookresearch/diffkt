/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package examples.hookeanSpring.simple

import io.kotest.core.spec.style.AnnotationSpec
import org.diffkt.*
import testutils.shouldBeExactly

class HookeanSpringTest : AnnotationSpec() {
    @Test fun testHookeanSpring() {
        val springs = SpringSystem.DEFAULT
        var vertices: DTensor = springs.initVertices
        repeat(100) {
            vertices = springs.update(vertices, 0.005f)
        }

        val expected =
            tensorOf(
                0.019129606F, -0.07035651F,
                0.92901534F, -0.13748522F,
                2.263584F, 0.52372885F,
                0.6655955F, 1.1077063F,
                1.6013515F, 1.9818759F,
                1.5877932F, -1.0728106F,
                0.46223664F, -0.9769208F,
                0.971294F, -1.8557358F
            ).reshape(Shape(8,2))

        vertices shouldBeExactly expected

    }
}
