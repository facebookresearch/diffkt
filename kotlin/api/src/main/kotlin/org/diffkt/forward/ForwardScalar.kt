/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.forward

import org.diffkt.DScalar
import org.diffkt.DTensor
import org.diffkt.Operations
import org.diffkt.adOptimize.ForwardDifferentiable

/**
 * A differentiable dual scalar (for forward derivatives)
 */
@ForwardDifferentiable("tangent")
open class ForwardScalar protected constructor(
    primal: DScalar,
    derivativeID: ForwardDerivativeID
) : ForwardTensor(primal, derivativeID), DScalar {
    constructor(
        primal: DScalar,
        derivativeID: ForwardDerivativeID,
        tangent: DTensor
    ) : this(primal, derivativeID) {
        super.tangent = tangent
    }
    override val primal: DScalar
        get() = super.primal as DScalar

    override val operations: Operations
        get() = ForwardScalarOperations
}
