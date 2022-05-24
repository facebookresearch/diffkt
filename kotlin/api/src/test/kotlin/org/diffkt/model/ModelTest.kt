/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import io.kotest.core.annotation.Tags
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.shouldBe
import org.diffkt.Device
import org.diffkt.basePrimal
import org.diffkt.combineHash
import kotlin.random.Random

val random = Random(123)

class SmallModel(
    override val layers: List<Layer<*>> = listOf(Dense(4, 2, random), Dense(2, 1, random))
) : Model<SmallModel>() {
    override fun hashCode(): Int = combineHash("SmallModel")
    override fun equals(other: Any?): Boolean = other is SmallModel && other.layers == layers
    override fun withLayers(newLayers: List<Layer<*>>) = SmallModel(newLayers)
}

@Tags("Gpu")
class ModelGpuTest : AnnotationSpec() {
    @Test fun toGPU() {
        val model = SmallModel()
        val gpuModel = model.gpu()
        (gpuModel.layers[0] as Dense).w.basePrimal().device shouldBe Device.GPU
        (gpuModel.layers[1] as Dense).w.basePrimal().device shouldBe Device.GPU
    }

    @Test fun toGPUAndBack() {
        val model = SmallModel()
        val model2 = model.gpu().cpu()
        (model2.layers[0] as Dense).w.basePrimal().device shouldBe Device.CPU
        (model2.layers[1] as Dense).w.basePrimal().device shouldBe Device.CPU
    }
}

