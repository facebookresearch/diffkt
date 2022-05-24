/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.external.ExternalLib
import org.diffkt.reverse.ReverseTensor

internal fun identityGradientofSameKind(x: DTensor, halfShape: Shape = x.shape): DTensor {
    return x.operations.identityGradientOfSameKind(x, halfShape)
}

internal fun zeroOfSameKind(x: DTensor, shape: Shape = x.shape): DTensor {
    return x.operations.zeroOfSameKind(x, shape)
}

// This is useful for testing
fun primal(x: Float, f: (DScalar) -> DScalar): Float =
        (f(FloatScalar(x)) as FloatScalar).value

// Used in distributions which lives in examples package
fun broadcastAll(a: DTensor, b: DTensor): Pair<DTensor, DTensor> {
    return Broadcasting.broadcastToCommonShape(a, b)
}

fun broadcastAll(a: DTensor, b: DTensor, c: DTensor): Triple<DTensor, DTensor, DTensor> {
    return Broadcasting.broadcastToCommonShape(a, b, c)
}

fun DTensor.broadcastToShape(shape: Shape) = this.broadcastTo(shape)

data class TensorInfo(val data: FloatArray, val strides: IntArray, val offset: Int) {
    constructor(tensor: StridedFloatTensor) : this(tensor.data, tensor.strides, tensor.offset)
}

fun List<Int>.isPrefix(list: List<Int>): Boolean {
    val prefix = this
    if (prefix.size > list.size) return false
    for (i in prefix.indices)
        if (prefix[i] != list[i]) return false
    return true
}

fun DTensor.primal(derivativeID: DerivativeID): DTensor {
    var primal = this
    while (primal.derivativeID.sequence > derivativeID.sequence)
        primal = primal.primal
    return if (primal.derivativeID == derivativeID) primal.primal else primal
}

/**
 * Returns the primal value as a FloatTensor.
 *
 * This will fail for tracing tensors.
 * */
fun DTensor.basePrimal(): FloatTensor = primal(NoDerivativeID) as FloatTensor

/**
 * Returns the primal value as a FloatScalar.
 *
 * This will fail for tracing tensors.
 * */
fun DScalar.basePrimal(): FloatScalar = primal(NoDerivativeID) as FloatScalar

fun DScalar.primal(derivativeID: DerivativeID): DScalar {
    var primal = this
    while (primal.derivativeID.sequence > derivativeID.sequence)
        primal = primal.primal
    return if (primal.derivativeID == derivativeID) primal.primal else primal
}

val DTensor.allAxes: IntArray get() = IntArray(rank) { it }

val IntArray.product get() = run {
    var product = 1
    for (elem in this)
        product *= elem
    product
}

fun DTensor.expandToTangent(tangent: DTensor): DTensor {
    if (this.shape == tangent.shape) return this
    val ones = Shape(IntArray(tangent.shape.rank - this.shape.rank) { 1 })
    return this.reshape(this.shape + ones)
}

internal fun DTensor.expandAndBroadcastToTangent(tangent: DTensor): DTensor {
    if (this.shape == tangent.shape) return this
    val ones = Shape(IntArray(tangent.shape.rank - this.shape.rank) { 1 })
    return this.reshape(this.shape + ones).broadcastTo(tangent.shape)
}

internal fun IntArray.remove(pos: Int): IntArray = when (pos) {
    0 -> sliceArray(1..lastIndex)
    lastIndex -> sliceArray(0 until lastIndex)
    else -> sliceArray(0 until pos) + sliceArray(pos + 1..lastIndex)
}

private fun shouldSendToCppImpl(sizeThreshold: Int, t: FloatTensor, checkLayout: Boolean = true, checkOffset: Boolean = true): Boolean {
    val sizeOK = t.size >= sizeThreshold
    val rankOK = t.rank <= 10
    val isStrided = (t is StridedFloatTensor
            && (t.offset == 0 || !checkOffset)
            && t.strides.all { it >= 0 })
    fun isNatural() = (t is StridedFloatTensor
            && t.layout == StridedUtils.Layout.NATURAL)

    return if (checkLayout)
        sizeOK && rankOK && isStrided && isNatural()
    else
        sizeOK && rankOK && isStrided
}

internal fun shouldSendToCpp(sizeThreshold: Int, t: FloatTensor, extern: ExternalLib, checkLayout: Boolean = true, checkOffset: Boolean = true): Boolean {
    if (!extern.isLoaded)
        return false
    return shouldSendToCppImpl(sizeThreshold, t, checkLayout, checkOffset)
}

internal fun shouldSendToCpp(sizeThreshold: Int, extern: ExternalLib, vararg tensors: FloatTensor, checkLayout: Boolean = true, checkOffset: Boolean = true): Boolean {
    if (!extern.isLoaded)
        return false
    for (t in tensors) {
        if (!shouldSendToCppImpl(sizeThreshold, t, checkLayout, checkOffset))
            return false
    }
    return true
}

/** Returns whether x is a constant or a single-reverse-derivative tensor */
internal fun isDnnlEligible(x: DTensor): Boolean {
    return x is FloatTensor ||
            (x is ReverseTensor && x.primal is FloatTensor)
}

/** Return the product of all vals in the list, or 1 if the list if empty. */
fun List<Int>.product(): Int {
    return this.fold(1, Int::times)
}

fun pr(lst: List<Int>): String = "[${lst.joinToString(", ") { it.toString() }}]"
fun pr(lst: IntArray): String = "[${lst.joinToString(", ") { it.toString() }}]"

fun combineHash(vararg values: Any?): Int {
    var result = 0
    for (v in values) {
        result = (result * 101) + v.hashCode()
    }
    return result
}

class LazyList<T : Any>(override val size: Int, val gen: (Int) -> T) : AbstractList<T>() {
    override fun get(index: Int): T {
        if (index < 0 || index >= size) throw IndexOutOfBoundsException("index $index is not in (0 until $size).")
        return gen(index)
    }
}