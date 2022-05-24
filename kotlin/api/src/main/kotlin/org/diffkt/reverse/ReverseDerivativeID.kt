/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.reverse

import org.diffkt.*
import org.diffkt.ActiveVariableReverseScalar
import org.diffkt.ActiveVariableReverseTensor
import java.util.*

class ReverseDerivativeID : DerivativeID() {
    private var savedUpstreamShape: Shape? = null
    var upstreamShape: Shape
        get() = savedUpstreamShape!!
        set(value) { assert(savedUpstreamShape == null); savedUpstreamShape = value }

    private val backpropagateWorkList: Stack<ReverseTensor> = Stack()
    fun addNode(t: ReverseTensor) = backpropagateWorkList.add(t)
    fun reversePass(): Map<ReverseTensor, DTensor> {
        val result = HashMap<ReverseTensor, DTensor>()
        while (!backpropagateWorkList.empty()) {
            val node = backpropagateWorkList.pop()
            node.backpropagate()
            if (node is ActiveVariableReverseTensor || node is ActiveVariableReverseScalar) {
                val upstream = node.upstream
                // TODO: should be assert, but those for some reason appear to be disabled during development
                require(upstream.shape == node.shape + upstreamShape)
                result.put(node, upstream)
            }
            node._savedUpstream = null
        }
        return result
    }
}