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
import shapeTyping.annotations.SType

/**
 * A forward differentiable tensor.
 */
@SType("S: Shape")
open class ForwardTensor protected constructor(
    override val primal: @SType("S") DTensor,
    override val derivativeID: ForwardDerivativeID,
) : DTensor {
    @SType("S: Shape")
    protected constructor(
        primal: @SType("S") DTensor,
        derivativeID: ForwardDerivativeID,
        // if derivativeID.inputTangentShapeForJacobian is non-empty,
        // tangent is treated as a batch of tangents to compute the Jacobian in one shot
        // starting from a batch of one-hot output tangents / identity tensor
        tangent: DTensor,
        @Suppress("UNUSED_PARAMETER") distinguisher: Int // distinguish from the problem function that would otherwise have the same signature
    ) : this(primal, derivativeID) {
        this.tangent = tangent
    }

    init {
        assert(derivativeID.sequence > primal.derivativeID.sequence)
    }

    private var savedTangent: DTensor? = null
    open var tangent: DTensor
        get() {
            return savedTangent!!
        }
        set(value) {
            assert(savedTangent == null)
            savedTangent = value
            assert(derivativeID.sequence > value.derivativeID.sequence)
            assert(primal.shape + derivativeID.inputTangentShapeForJacobian == value.shape)
        }

    override val operations: Operations
        get() = ForwardTensorOperations

    override fun toCodeString(): String = "(${primal.toCodeString()} + $derivativeID ${tangent.toCodeString()})"
    override fun toString(): String = "($primal + $derivativeID $tangent)"

    companion object {
        @SType("S: Shape")
        operator fun invoke(
            primal: @SType("S") DTensor,
            derivativeID: ForwardDerivativeID,
            tangent: DTensor,
        ): @SType("S") ForwardTensor {
            return if (primal is DScalar)
                ForwardScalar(primal, derivativeID, tangent)
            else
                ForwardTensor(primal, derivativeID, tangent, 0)
        }
    }
}
