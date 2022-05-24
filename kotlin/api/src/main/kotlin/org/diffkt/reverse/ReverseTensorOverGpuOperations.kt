/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import org.diffkt.*
import org.diffkt.external.Gpu
import org.diffkt.gpu.GpuFloatScalar
import org.diffkt.gpu.GpuFloatTensor
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

@AllowUnreduced
internal object ReverseTensorOverGpuOperations: ReverseTensorOperationsImpl() {
    override val name get() = "ReverseTensorOverGpu"

    @SType("S: Shape")
    private fun wrap(value: @SType("S") DTensor, derivativeId: ReverseDerivativeID): @SType("S") ReverseTensor {
        if (value is ReverseTensor && value.derivativeID.sequence == derivativeId.sequence && value.primal is GpuFloatTensor)
            return value
        require(value.derivativeID.sequence < derivativeId.sequence)
        require(value is GpuFloatTensor) // Necessitates value !is DScalar
        return object : ReverseTensor(primal = value, derivativeID = derivativeId) {
            override fun backpropagate() { }
        }
    }

    @SType("S: Shape")
    override fun plus(
            left: @SType("S") DTensor,
            right: @SType("S") DTensor,
            derivativeId: DerivativeID
    ): @SType("S") ReverseTensor {
        require(derivativeId is ReverseDerivativeID)
        val l = wrap(left, derivativeId)
        val r = wrap(right, derivativeId)
        val lp = l.primal as GpuFloatTensor
        val rp = r.primal as GpuFloatTensor
        require(!l.isScalar)

        val detachedAndResHandles = Gpu.add(lp.handle, rp.handle)
        val detachedLhs = GpuFloatTensor(detachedAndResHandles[0])
        val detachedRhs = GpuFloatTensor(detachedAndResHandles[1])
        val newPrimal = GpuFloatTensor(detachedAndResHandles[2])
        return object : ReverseTensor(newPrimal, l.derivativeID) {
            override fun backpropagate() {
                require(derivativeID.upstreamShape == Shape()) {
                    "GPU gradients are only supported for functions that return a scalar" }
                val upstreamT = upstream
                require(upstreamT is FloatTensor) { "Higher order GPU gradients not supported" }
                require(upstreamT is GpuFloatTensor) { "Upstream must be a GPU tensor, was ${upstreamT::class}" }
                l.pushback(GpuFloatTensor(Gpu.addGradLhs(upstreamT.handle, detachedLhs.handle, newPrimal.handle)))
                r.pushback(GpuFloatTensor(Gpu.addGradRhs(upstreamT.handle, detachedRhs.handle, newPrimal.handle)))
            }
        }
    }

    override fun matmul(
        x: DTensor,
        y: DTensor,
        a: Shape,
        b: Shape,
        c: Shape,
        d: Shape,
        derivativeId: DerivativeID
    ): DTensor {
        require(derivativeId is ReverseDerivativeID)
        val l = wrap(x, derivativeId)
        val r = wrap(y, derivativeId)
        val lp = l.primal as GpuFloatTensor
        val rp = r.primal as GpuFloatTensor

        val detachedAndResHandles = Gpu.matmul(lp.handle, rp.handle)
        val detachedLhs = GpuFloatTensor(detachedAndResHandles[0])
        val detachedRhs = GpuFloatTensor(detachedAndResHandles[1])
        val newPrimal = GpuFloatTensor(detachedAndResHandles[2])
        return object : ReverseTensor(newPrimal, l.derivativeID) {
            override fun backpropagate() {
                require(derivativeID.upstreamShape == Shape()) {
                    "GPU gradients are only supported for functions that return a scalar"
                }
                val upstreamT = upstream
                require(upstreamT is FloatTensor) {
                    "Higher order GPU gradients not supported"
                }
                require(upstreamT is GpuFloatTensor) {
                    "Upstream must be a GPU tensor, was ${upstreamT::class}"
                }
                l.pushback(GpuFloatTensor(Gpu.matmulGradLhs(upstreamT.handle, detachedLhs.handle, newPrimal.handle)))
                r.pushback(GpuFloatTensor(Gpu.matmulGradRhs(upstreamT.handle, detachedRhs.handle, newPrimal.handle)))
            }
        }
    }

    override fun broadcastTo(x: DTensor, newShape: Shape): DTensor {
        require(x is ReverseTensor)
        val xp = x.primal
        require(xp is GpuFloatTensor) {
            "Only single derivatives are supported on GPU"
        }
        val detachedAndResHandles = Gpu.broadcastTo(xp.handle, newShape.dims)
        val detachedX = GpuFloatTensor(detachedAndResHandles[0])
        val newPrimal = GpuFloatTensor(detachedAndResHandles[1])

        return object : ReverseTensor(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                require(derivativeID.upstreamShape == Shape()) {
                    "GPU gradients are only supported for functions that return a scalar"
                }
                val upstreamT = upstream
                require(upstreamT is FloatTensor) {
                    "Higher order GPU gradients not supported"
                }
                require(upstreamT is GpuFloatTensor) {
                    "Upstream must be a GPU tensor, was ${upstreamT::class}"
                }
                x.pushback(GpuFloatTensor(Gpu.broadcastToGrad(upstreamT.handle, detachedX.handle, newPrimal.handle)))
            }
        }
    }

    @SType("S: Shape")
    override fun relu(x: @SType("S") DTensor): @SType("S") DTensor {
        require(x is ReverseTensor)
        val xp = x.primal
        require(xp is GpuFloatTensor) {
            "Only single derivatives are supported on GPU"
        }
        val detachedAndResHandles = Gpu.relu(xp.handle)
        val detachedX = GpuFloatTensor(detachedAndResHandles[0])
        val newPrimal = GpuFloatTensor(detachedAndResHandles[1])
        return object : ReverseTensor(newPrimal, x.derivativeID) {
            override fun backpropagate() {
                require(derivativeID.upstreamShape == Shape()) {
                    "GPU gradients are only supported for functions that return a scalar"
                }
                val upstreamT = upstream
                require(upstreamT is FloatTensor) {
                    "Higher order GPU gradients not supported"
                }
                require(upstreamT is GpuFloatTensor) {
                    "Upstream must be a GPU tensor, was ${upstreamT::class}"
                }
                x.pushback(GpuFloatTensor(Gpu.reluGrad(upstreamT.handle, detachedX.handle, newPrimal.handle)))
            }
        }
    }

    override fun sum(x: DTensor, axes: IntArray, keepDims: Boolean): DTensor {
        require(x is ReverseTensor)
        val xp = x.primal
        require(xp is GpuFloatTensor) {
            "Only single derivatives are supported on GPU"
        }
        val detachedAndresHandles = Gpu.sum(xp.handle, axes, keepDims)
        val detachedX = GpuFloatTensor(detachedAndresHandles[0])
        val resShape = Gpu.getShape(detachedAndresHandles[1])
        val newPrimal = if (resShape.contentEquals(intArrayOf())) GpuFloatScalar(detachedAndresHandles[1]) else GpuFloatTensor(detachedAndresHandles[1])

        return if (newPrimal is GpuFloatScalar)
            object : ReverseScalar(GpuFloatScalar(detachedAndresHandles[1]), x.derivativeID)  {
                override fun backpropagate() {
                    require(derivativeID.upstreamShape == Shape()) {
                        "GPU gradients are only supported for functions that return a scalar"
                    }
                    val upstreamT = upstream
                    require(upstreamT is FloatTensor) {
                        "Higher order GPU gradients not supported"
                    }
                    require(upstreamT is GpuFloatTensor) {
                        "Upstream must be a GPU tensor, was ${upstreamT::class}"
                    }
                    x.pushback(GpuFloatTensor(Gpu.sumGrad(upstreamT.handle, detachedX.handle, newPrimal.handle)))
                }

            }
        else
            object : ReverseTensor(GpuFloatTensor(detachedAndresHandles[1]), x.derivativeID) {
            override fun backpropagate() {
                require(newPrimal is GpuFloatTensor)
                require(derivativeID.upstreamShape == Shape()) {
                    "GPU gradients are only supported for functions that return a scalar"
                }
                val upstreamT = upstream
                require(upstreamT is FloatTensor) {
                    "Higher order GPU gradients not supported"
                }
                require(upstreamT is GpuFloatTensor) {
                    "Upstream must be a GPU tensor, was ${upstreamT::class}"
                }
                x.pushback(GpuFloatTensor(Gpu.sumGrad(upstreamT.handle, detachedX.handle, newPrimal.handle)))
            }
        }
    }
}
