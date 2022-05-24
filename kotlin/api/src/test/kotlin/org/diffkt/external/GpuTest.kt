/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.external

import io.kotest.core.annotation.Tags
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe
import org.diffkt.*
import testutils.*

@Tags("Gpu")
class GpuTest : AnnotationSpec() {
    @Test fun putGetAndDelete() {
        val shape = Shape(2, 3)
        val data = floats(6)

        val handle = Gpu.putFloatTensor(shape.dims, data)
        val fetchedShape = Gpu.getShape(handle)
        val fetchedData = Gpu.getFloatData(handle)
        Gpu.deleteHandle(handle)

        fetchedShape shouldBe shape.dims
        fetchedData shouldBe data
    }
}
