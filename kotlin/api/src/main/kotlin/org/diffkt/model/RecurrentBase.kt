/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.model

import org.diffkt.*

interface RecurrentBase<Recurrent : RecurrentBase<Recurrent, T>, T> : TrainableLayer<Recurrent> {
    val initialState: T
    val initialOutput: DTensor
    val batchAxis: Int get() = 0
    val sequenceAxis: Int get() = 1

    val accType: AccType

    // Process a batch against current hidden state,
    // returning a new hidden state and an output tensor.
    // This is used in a fold over an input sequence. Signature:
    // ((prevHidden, prevOutput), element) => (newHidden, newOutput)
    fun cell(state: Pair<T, DTensor>, x: DTensor): Pair<T, DTensor>

    // Not super-friendly that we're asking the user to provide this (in most cases it's just stretching hidden out to
    // batch size. But since hidden is generic we don't have a good way around it right now.
    fun processForBatching(
        initialState: T,
        initialOutput: DTensor,
        batchSize: Int
    ): Pair<T, DTensor>

    // RNN as a fold type is explained here:
    // https://colah.github.io/posts/2015-09-NN-Types-FP/
    fun fold(t: DTensor, sequenceAxis: Int, initialState: T): Pair<T, DTensor> {
        return if (t.size == 0) {
            Pair(initialState, t)
        } else {
            // Unroll the first loop iteration to get a starting output with correct shape
            var slice = t.axisIdx(0, sequenceAxis).squeeze(sequenceAxis)
            val (batchedHidden, batchedOutput) = processForBatching(initialState, initialOutput, t.shape[batchAxis])
            var result = cell(Pair(batchedHidden, batchedOutput), slice)

            for (i in 1 until t.shape[sequenceAxis]) {
                slice = t.axisIdx(i, sequenceAxis).squeeze(sequenceAxis)
                result = cell(result, slice)
            }
            result
        }
    }

    /** Returns a one-element-wide slice of the given axis at the given index */
    private fun DTensor.axisIdx(n: Int, axis: Int = 0): DTensor = this.slice(n, n + 1, axis)

    // Generalized RNN (with output per input slice) as an accumulating map is explained here:
    // https://colah.github.io/posts/2015-09-NN-Types-FP/
    // This has the same type as fold since Tensor is so all-encompassing, but the output tensor is more like Seq[Tensor]
    // (or a Tensor of one greater dimension)
    fun accMap(t: DTensor, sequenceAxis: Int, initialState: T): Pair<T, DTensor> {
        return if (t.size == 0) {
            Pair(initialState, t)
        } else {
            val shape = t.shape
            val results = mutableListOf<DTensor>()
            var r = processForBatching(initialState, initialOutput, t.shape[batchAxis])

            for (i in 0 until shape[sequenceAxis]) {
                val slice = t.axisIdx(i, sequenceAxis).squeeze(sequenceAxis)
                r = cell(r, slice)
                results += r.second.unsqueeze(sequenceAxis)
            }
            Pair(r.first, concat(results, sequenceAxis))
        }
    }

    override fun invoke(vararg inputs: DTensor): DTensor {
        require(inputs.size in 1..2) { "Expected 1 or 2 inputs, got ${inputs.size} " }
        return if (inputs.size == 1)
            doRecurrence(inputs[0])
        else {
            // This cast is unfortunate but if T is DTensor then I can't seem
            // to make doRecurrence (when renamed to invoke) be the one called.
            // TODO: https://github.com/facebookincubator/diffkt/issues/226
            // TODO: revisit and clean this up, perhaps when we have an LSTM hooked up.
            // TODO: alternatively, don't allow the initial hidden state to be passed
            //   here (only at layer creation time)
            @Suppress("UNCHECKED_CAST")
            doRecurrence(inputs[0], inputs[1] as T)
        }
    }

    /**
     * Do the recurrence.
     *
     * @param x a tensor of shape (batch size, sequence length, num inputs)
     * @param initialState (optional) the initial state. Defaults to
     *     this.initialState
     * @return If AccType is Fold, returns a tensor of shape (batchSize, num outputs);
     *     If AccType is AccMap, returns a tensor of shape (batchSize, sequence length,
     *     num outputs).
     */
    fun doRecurrence(x: DTensor, initialState: T = this.initialState): DTensor {
        require(x.rank == 3) { "input must be rank 3, got rank ${x.rank} " }
        val result = when (accType) {
            AccType.Fold -> fold(x, sequenceAxis, initialState)
            AccType.AccMap -> accMap(x, sequenceAxis, initialState)
        }
        return result.second
    }

    companion object RecurrentBase {
        enum class AccType {
            Fold,
            AccMap
        }
    }
}
