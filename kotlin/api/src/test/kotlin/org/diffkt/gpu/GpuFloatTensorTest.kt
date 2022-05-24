/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.gpu

import io.kotest.assertions.throwables.shouldThrow
import io.kotest.core.annotation.Tags
import io.kotest.core.spec.style.AnnotationSpec
import io.kotest.matchers.string.shouldContain
import io.kotest.matchers.shouldBe
import org.diffkt.*
import kotlin.random.Random

@Tags("Gpu")
class GpuFloatTensorTest : AnnotationSpec() {

    @Test
    fun device() {
        val t = FloatTensor.random(Random(123), Shape(2))
        val s = FloatScalar(42f)

        t.gpu().cpu() shouldBe t
        s.gpu().cpu() shouldBe s

        t.to(Device.CPU) shouldBe t
        t.to(Device.GPU).to(Device.CPU) shouldBe t
        s.to(Device.CPU) shouldBe s
        s.to(Device.GPU).to(Device.CPU) shouldBe s

        t.device shouldBe Device.CPU
        s.device shouldBe Device.CPU
        t.gpu().device shouldBe Device.GPU
        s.gpu().device shouldBe Device.GPU
        t.to(Device.GPU).device shouldBe Device.GPU
        s.to(Device.GPU).device shouldBe Device.GPU
    }

    @Test
    fun primalAndPullbackPlus() {
        val cpuX = FloatTensor.random(Random(123), Shape(2, 3))
        val (cpuPrimal, cpuGrad) = primalAndVjp(cpuX, { cpuPrimal -> FloatTensor.ones(cpuPrimal.shape) }, ::add)

        val x = cpuX.gpu()
        val (primal, grad) = primalAndVjp(x, { primal -> FloatTensor.ones(primal.shape).gpu() }, ::add)

        (primal as GpuFloatTensor).cpu() shouldBe cpuPrimal
        (grad as GpuFloatTensor).cpu() shouldBe cpuGrad
    }

    @Test
    fun gpuPullbackCalledWithCpuArgument() {
        val x = FloatTensor.random(Random(123), Shape(2, 3)).gpu()

        val e = shouldThrow<IllegalArgumentException> {
            primalAndVjp(x, { primal -> FloatTensor.ones(primal.shape) }, ::add)
        }
        e.message shouldContain "Upstream must be a GPU tensor"
    }

    @Test
    fun primalAndReverseDerivativeScalar() {
        val cpuX = FloatTensor.random(Random(123), Shape())
        val gpuX = cpuX.gpu()
        val (gpuPrimal, gpuGrad) = primalAndReverseDerivative(gpuX, ::add)

        val (cpuPrimal, cpuGrad) = primalAndReverseDerivative(cpuX, ::add)
        (gpuPrimal as GpuFloatTensor).cpu() shouldBe cpuPrimal
        (gpuGrad as GpuFloatTensor).cpu() shouldBe cpuGrad
    }

    @Test
    fun primalAndReverseDerivativeTensor() {
        val gpuTensor = FloatTensor.random(Random(123), Shape(2, 3)).gpu()
        val e = shouldThrow<IllegalArgumentException> { primalAndReverseDerivative(gpuTensor, ::add) }
        e.message shouldContain "GPU gradients are only supported for functions that return a scalar"
    }

    @Test
    fun primalAndForwardDerivativeScalar() {
        val cpuX = FloatTensor.random(Random(123), Shape())
        val gpuX = cpuX.gpu()
        val (gpuPrimal, gpuGrad) = primalAndForwardDerivative(gpuX, ::add)

        val (cpuPrimal, cpuGrad) = primalAndForwardDerivative(cpuX, ::add)
        (gpuPrimal as GpuFloatTensor).cpu() shouldBe cpuPrimal
        (gpuGrad as GpuFloatTensor).cpu() shouldBe cpuGrad
    }

    @Test
    fun primalAndForwardDerivativeTensor() {
        val cpuX = FloatTensor.random(Random(123), Shape(2, 3))
        val gpuX = cpuX.gpu()
        val (gpuPrimal, gpuGrad) = primalAndForwardDerivative(gpuX, ::add)

        val (cpuPrimal, cpuGrad) = primalAndForwardDerivative(cpuX, ::add)
        (gpuPrimal as GpuFloatTensor).cpu() shouldBe cpuPrimal
        (gpuGrad as GpuFloatTensor).cpu() shouldBe cpuGrad
    }

    fun add(x: DTensor): DTensor {
        return x + x
    }
}
