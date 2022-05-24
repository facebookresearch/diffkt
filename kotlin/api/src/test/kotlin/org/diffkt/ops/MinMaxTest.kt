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

class MinMaxTest: AnnotationSpec() {

    private val t4x4 = tensorOf(
        2f, 3f, 4f, 1f,
        6f, 20f, 0f, 9f,
        16f, 1f, 3f, 4f,
        10f, 2f, 4f, 1f
    ).reshape(4,4) as FloatTensor

    private val t3x3x3 = tensorOf(
        2f, 6f, 8f,
        4f, 0f, 3f,
        11f, 9f, 2f,

        2f, 2f, 2f,
        1f, 3f, 4f,
        14f, 5f, 1f,

        0f, 4f, 7f,
        6f, 18f, 5f,
        3f, 12f, 2f
    ).reshape(3,3,3) as FloatTensor

    @Test
    fun maxAllAxes2D() {
        t4x4.max() shouldBe FloatScalar(20f)
        t4x4.max(keepDims = true) shouldBe tensorOf(20f).reshape(1,1)
    }

    @Test
    fun minAllAxes2D() {
        t4x4.min() shouldBe FloatScalar(0f)
        t4x4.min(keepDims = true) shouldBe tensorOf(0f).reshape(1,1)
    }

    @Test
    fun maxAllAxes3D() {
        t3x3x3.max() shouldBe FloatScalar(18f)
        t3x3x3.max(keepDims = true) shouldBe tensorOf(18f).reshape(1,1,1)
    }

    @Test
    fun minAllAxes3D() {
        t3x3x3.min() shouldBe FloatScalar(0f)
        t3x3x3.min(keepDims = true) shouldBe tensorOf(0f).reshape(1,1,1)
    }

    @Test
    fun maxSingleAxis2D() {
        val resultValues0 = tensorOf(16f, 20f, 4f, 9f)
        val resultValues1 = tensorOf(4f, 20f, 16f, 10f)

        t4x4.max(intArrayOf(0)) shouldBe resultValues0
        t4x4.max(intArrayOf(0), keepDims = true) shouldBe resultValues0.reshape(1,4)

        t4x4.max(intArrayOf(1)) shouldBe resultValues1
        t4x4.max(intArrayOf(1), keepDims = true) shouldBe resultValues1.reshape(4,1)
    }

    @Test
    fun minSingleAxis2D() {
        val resultValues0 = tensorOf(2f, 1f, 0f, 1f)
        val resultValues1 = tensorOf(1f, 0f, 1f, 1f)

        t4x4.min(intArrayOf(0)) shouldBe resultValues0
        t4x4.min(intArrayOf(0), keepDims = true) shouldBe resultValues0.reshape(1,4)

        t4x4.min(intArrayOf(1)) shouldBe resultValues1
        t4x4.min(intArrayOf(1), keepDims = true) shouldBe resultValues1.reshape(4,1)
    }

    @Test
    fun maxSingleAxis3D() {
        val resultValues0 = tensorOf(2f, 6f, 8f, 6f, 18f, 5f, 14f, 12f, 2f)
        val resultValues1 = tensorOf(11f, 9f, 8f, 14f, 5f, 4f, 6f, 18f, 7f)
        val resultValues2 = tensorOf(8f, 4f, 11f, 2f, 4f, 14f, 7f, 18f, 12f)

        t3x3x3.max(intArrayOf(0)) shouldBe resultValues0.reshape(3,3)
        t3x3x3.max(intArrayOf(1)) shouldBe resultValues1.reshape(3,3)
        t3x3x3.max(intArrayOf(2)) shouldBe resultValues2.reshape(3,3)

        t3x3x3.max(intArrayOf(0), keepDims = true) shouldBe resultValues0.reshape(1,3,3)
        t3x3x3.max(intArrayOf(1), keepDims = true) shouldBe resultValues1.reshape(3,1,3)
        t3x3x3.max(intArrayOf(2), keepDims = true) shouldBe resultValues2.reshape(3,3,1)
    }


    @Test
    fun minSingleAxis3D() {
        val resultValues0 = tensorOf(0f, 2f, 2f, 1f, 0f, 3f, 3f, 5f, 1f)
        val resultValues1 = tensorOf(2f, 0f, 2f, 1f, 2f, 1f, 0f, 4f, 2f)
        val resultValues2 = tensorOf(2f, 0f, 2f, 2f, 1f, 1f, 0f, 5f, 2f)

        t3x3x3.min(intArrayOf(0)) shouldBe resultValues0.reshape(3,3)
        t3x3x3.min(intArrayOf(1)) shouldBe resultValues1.reshape(3,3)
        t3x3x3.min(intArrayOf(2)) shouldBe resultValues2.reshape(3,3)

        t3x3x3.min(intArrayOf(0), keepDims = true) shouldBe resultValues0.reshape(1,3,3)
        t3x3x3.min(intArrayOf(1), keepDims = true) shouldBe resultValues1.reshape(3,1,3)
        t3x3x3.min(intArrayOf(2), keepDims = true) shouldBe resultValues2.reshape(3,3,1)
    }

    @Test
    fun max2of3Axes() {
        val result01 = tensorOf(14f, 18f, 8f)
        val result02 = tensorOf(8f, 18f, 14f)
        val result12 = tensorOf(11f, 14f, 18f)

        t3x3x3.max(intArrayOf(0,1)) shouldBe result01
        t3x3x3.max(intArrayOf(0,1), keepDims = true) shouldBe result01.reshape(1,1,3)

        t3x3x3.max(intArrayOf(0,2)) shouldBe result02
        t3x3x3.max(intArrayOf(0,2), keepDims = true) shouldBe result02.reshape(1,3,1)

        t3x3x3.max(intArrayOf(1,2)) shouldBe result12
        t3x3x3.max(intArrayOf(1,2), keepDims = true) shouldBe result12.reshape(3,1,1)
    }

    @Test
    fun min2of3Axes() {
        val result01 = tensorOf(0f, 0f, 1f)
        val result02 = tensorOf(0f, 0f, 1f)
        val result12 = tensorOf(0f, 1f, 0f)

        t3x3x3.min(intArrayOf(0,1)) shouldBe result01
        t3x3x3.min(intArrayOf(0,1), keepDims = true) shouldBe result01.reshape(1,1,3)

        t3x3x3.min(intArrayOf(0,2)) shouldBe result02
        t3x3x3.min(intArrayOf(0,2), keepDims = true) shouldBe result02.reshape(1,3,1)

        t3x3x3.min(intArrayOf(1,2)) shouldBe result12
        t3x3x3.min(intArrayOf(1,2), keepDims = true) shouldBe result12.reshape(3,1,1)
    }

}
