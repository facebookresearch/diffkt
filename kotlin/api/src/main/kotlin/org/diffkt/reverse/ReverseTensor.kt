/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import org.diffkt.*
import org.diffkt.gpu.GpuFloatTensor
import org.diffkt.zeroOfSameKind
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

/**
 * A tensor for reverse mode differentiation.
 */
@SType("S: Shape")
abstract class ReverseTensor(
    override val primal: @SType("S") DTensor,
    override final val derivativeID: ReverseDerivativeID,
) : @SType("S") DTensor {
    init {
        assert(derivativeID.sequence > 0)
        derivativeID.addNode(this)
    }

    internal var _savedUpstream: DTensor? = null
    var upstream: DTensor
        get() {
            if (_savedUpstream == null)
                _savedUpstream = zeroOfSameKind(primal, primal.shape + derivativeID.upstreamShape)
            return _savedUpstream!!
        }
        set(value) {
            require(value.shape == primal.shape + derivativeID.upstreamShape)
            _savedUpstream = value
        }

    internal val hasUpstream: Boolean
        get() = _savedUpstream != null

    fun pushback(upstream: DTensor) {
        require(upstream.derivativeID.sequence < derivativeID.sequence)
        val oldUpstream = _savedUpstream
        val newUpstream = if (oldUpstream == null)
            upstream
        else
            oldUpstream + upstream
        this.upstream = newUpstream
    }

    abstract fun backpropagate()
    override val operations: Operations
        get() = if (primal is GpuFloatTensor) ReverseTensorOverGpuOperations else ReverseTensorOperations

    override fun toCodeString(): String = "(${primal.toCodeString()} + $derivativeID})"
    override fun toString(): String = "($primal + $derivativeID)"
}
