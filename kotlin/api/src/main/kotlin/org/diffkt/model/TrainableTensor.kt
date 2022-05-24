/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*
import java.nio.ByteBuffer
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

@SType("S: Shape")
@AllowUnreduced
class TrainableTensor(val tensor: @SType("S") DTensor) : Trainable<@SType("S") TrainableTensor> {
    override fun trainingStep(optim: Optimizer<*>, tangent: Trainable.Tangent): @SType("S") TrainableTensor {
        require(tangent is Tangent)
        assert(tangent.value.shape == tensor.shape) { "Cannot adjust(): tangent shape does not match primal shape" }
        val newTensor = optim.tensorTrainingStep(tensor, tangent.value)
        return TrainableTensor(newTensor)
    }

    override fun wrap(wrapper: Wrapper) = TrainableTensor(wrapper.wrapDTensor(tensor))

    override fun store(into: ByteBuffer): ByteBuffer {
        val tensor = this.tensor as FloatTensor
        val spaceNeeded = (
                1 // rank
                + tensor.rank // each dim
                + tensor.size // each datum
                ) * 4;
        val sink = if (into.capacity() - into.position() < spaceNeeded) {
            val expandSize = Math.max((into.limit() * 2), into.position() + spaceNeeded)
            val expandedBuffer = ByteBuffer.allocate(expandSize)
            into.flip()
            expandedBuffer.put(into)
            expandedBuffer
        } else into
        val s1 = sink.asIntBuffer()
        s1.put(tensor.rank)
        for (d in tensor.shape.dims) s1.put(d)
        sink.position(sink.position() + (1 + tensor.rank) * 4)
        val s2 = sink.asFloatBuffer()
        for (pos in 0 until tensor.size) {
            s2.put(tensor.at(pos))
        }
        sink.position(sink.position() + tensor.size * 4)
        return sink
    }

    override fun load(from: ByteBuffer): TrainableTensor {
        val rank = from.int
        val shape = Shape(IntArray(rank) { from.int })
        require(shape == this.tensor.shape)
        val size = shape.product
        val data = FloatArray(size)
        from.asFloatBuffer().get(data)
        from.position(from.position() + size * 4)
        val tensor = FloatTensor(shape, data)
        return TrainableTensor(tensor)
    }

    override fun extractTangent(
        output: DTensor,
        extractor: (input: DTensor, output: DTensor) -> DTensor
    ): Tangent {
        return Tangent(extractor(tensor, output))
    }

    override fun cpu(): TrainableTensor {
        if (tensor !is FloatTensor) TODO("Handle fancy differentiation")
        return TrainableTensor(tensor.cpu())
    }

    override fun gpu(): TrainableTensor {
        if (tensor !is FloatTensor) TODO("Handle fancy differentiation")
        return TrainableTensor(tensor.gpu())
    }

    override fun hashCode(): Int = combineHash("TrainableTensor", tensor)
    override fun equals(other: Any?): Boolean = other is TrainableTensor &&
            other.tensor == tensor

    companion object {
        data class Tangent(val value: DTensor) : Trainable.Tangent
    }
}
