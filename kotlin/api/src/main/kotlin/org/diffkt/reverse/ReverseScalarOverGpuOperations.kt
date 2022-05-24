/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import org.diffkt.DScalar
import org.diffkt.DTensor
import org.diffkt.gpu.GpuFloatScalar

internal object ReverseScalarOverGpuOperations: ReverseTensorOperationsImpl() {
    override val name get() = "ReverseScalarOverGpu"
    private fun wrap(value: DTensor, derivativeId: ReverseDerivativeID): ReverseScalar {
        require(value is DScalar)
        if (value is ReverseScalar && value.derivativeID == derivativeId && value.primal is GpuFloatScalar)
            return value
        require(value.derivativeID.sequence < derivativeId.sequence)
        require(value is GpuFloatScalar)
        return object : ReverseScalar(primal = value, derivativeID = derivativeId) {
            override fun backpropagate() { }
        }
    }

}