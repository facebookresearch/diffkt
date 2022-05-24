/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.external.Dnnl

/**
 * x is the output of the network: x[i,j] is its guess on whether input i is classified as a j.
 * labels is the actual label: labels[i,j] is 1 iff input i is classified as a j; 0 otherwise.
 */
fun crossEntropyLossFromOneHot(x: DTensor, oneHotLabels: DTensor) : DScalar {
    return nllLossFromOneHot(x.logSoftmax(axis = 1), oneHotLabels)
}

/**
 * x is the output of the network: x[i,j] is its guess on whether input i is classified as a j.
 * labels is the actual label: labels[i,j] is 1 iff input i is classified as a j; 0 otherwise.
 */
fun crossEntropyLoss(x: DTensor, labels: DTensor, fromOneHot: Boolean = false) : DScalar {
    val oneHotLabels = if (fromOneHot) labels else createOneHotFromClasses(x, labels as FloatTensor)
    return crossEntropyLossFromOneHot(x, oneHotLabels)
}

/**
 * Creates a one hot tensor for nll loss calculation when provided l, a tensor of class IDs.
 */
private fun createOneHotFromClasses(x: DTensor, l: FloatTensor): FloatTensor {
    require(x.rank == 2)
    require(l.rank == 1)
    require(x.shape[0] == l.shape[0]) { "NLL weight and target dimension mismatch at dim 0 (weight: ${x.shape[0]}, target: ${l.shape[0]}" }

    val data = FloatArray(x.size)
    val nClasses = x.shape.last
    for (pos in 0 until l.size) {
        val classId = l.at(pos).toInt()
        val newPos = pos * nClasses + classId
        data[newPos] = 1f
    }

    return FloatTensor(x.shape, data)
}

fun DTensor.logSoftmax(axis: Int): DTensor {
    return this.operations.logSoftmax(this, axis)
}
internal fun baseLogSoftmax(x: DTensor, axis: Int): DTensor {
    val maxes = x.basePrimal().max(intArrayOf(axis), keepDims = true)
    val diffToMaxes = x - maxes
    return diffToMaxes - ln(exp(diffToMaxes).sum(axis, keepDims = true))
}

/**
 * Applies softmax along the provided axis, rescaling the tensor
 * so that every slice along the provided axis is in the range [0,1]
 * and sums to 1.
 *
 * Output shape is the same as the input shape
 *
 * Also see https://towardsdatascience.com/softmax-activation-function-how-it-actually-works-d292d335bd78
 */
fun softmax(x: DTensor, axis: Int): DTensor {
    val shifted = x - x.basePrimal().max(intArrayOf(axis), keepDims = true)
    val e = exp(shifted)
    val se = e.sum(intArrayOf(axis), keepDims = true)
    return e / se
}

@JvmName("DTensorSoftmaxExt")
fun DTensor.softmax(axis: Int): DTensor = softmax(this, axis)

fun nllLossFromOneHot(guesses: DTensor, labels: DTensor) : DScalar {
    val n = guesses.size / guesses.shape[1] // number of samples
    return - (guesses * labels).sum() / n.toFloat()
}
