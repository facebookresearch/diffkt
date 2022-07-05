/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import org.diffkt.DScalar
import org.diffkt.gpu.GpuFloatScalar
import org.diffkt.Operations
import org.diffkt.adOptimize.ReverseDifferentiable

/**
 * A scalar for reverse mode differentiation.
 */
@ReverseDifferentiable("primal", "upstream", "backpropagate", "pushback", "derivativeID")
abstract class ReverseScalar(override val primal: DScalar, derivativeID: ReverseDerivativeID) : ReverseTensor(primal, derivativeID),
    DScalar {
    override val operations: Operations
        get() = if (primal is GpuFloatScalar) ReverseScalarOverGpuOperations else ReverseScalarOperations
}
