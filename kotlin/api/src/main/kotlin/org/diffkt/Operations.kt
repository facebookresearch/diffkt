/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.Convolve.PaddingStyle
import org.diffkt.model.BatchNormResult
import org.diffkt.model.baseBatchNorm
import org.diffkt.random.RandomKey
import org.diffkt.tracing.TracingTensorOperations
import shapeTyping.annotations.AllowUnreduced
import shapeTyping.annotations.SType

@AllowUnreduced
interface Operations {
    val name: String

    /** For a scalar operations, returns the corresponding tensor operations. */
    val tensor: Operations get() = this

    @SType("S: Shape")
    fun plus(left: @SType("S") DTensor, right: @SType("S") DTensor, derivativeId: DerivativeID): @SType("S") DTensor

    @SType("S: Shape")
    fun minus(left: @SType("S") DTensor, right: @SType("S") DTensor, derivativeId: DerivativeID): @SType("S") DTensor

    @SType("S: Shape")
    fun times(left: @SType("S") DTensor, right: @SType("S") DTensor, derivativeId: DerivativeID): @SType("S") DTensor

    @SType("S: Shape")
    fun timesScalar(left: DScalar, right: @SType("S") DTensor, derivativeId: DerivativeID): @SType("S") DTensor

    @SType("S: Shape")
    fun div(left: @SType("S") DTensor, right: @SType("S") DTensor, derivativeId: DerivativeID): @SType("S") DTensor =
        left * right.pow(-1)

    @SType("S: Shape")
    fun zeroOfSameKind(x: DTensor, shape: @SType("S") Shape): @SType("S") DTensor

    @SType("S: Shape")
    fun identityGradientOfSameKind(x: DTensor, halfShape: @SType("S") Shape): @SType("concat(S,S)") DTensor

    @SType("S: Shape")
    fun unaryMinus(x: @SType("S") DTensor): @SType("S") DTensor

    fun matmul(x: DTensor, y: DTensor, a: Shape, b: Shape, c: Shape, d: Shape, derivativeId: DerivativeID): DTensor

    @SType("S1: Shape, S2: Shape")
    fun outerProduct(x: @SType("S1") DTensor, y: @SType("S2") DTensor, derivativeId: DerivativeID): @SType("concat(S1, S2)") DTensor

    @SType("S: Shape") fun sin(x: @SType("S") DTensor): DTensor
    @SType("S: Shape") fun cos(x: @SType("S") DTensor): DTensor
    @SType("S: Shape") fun tan(x: @SType("S") DTensor): DTensor
    @SType("S: Shape") fun atan(x: @SType("S") DTensor): DTensor
    @SType("S: Shape") fun exp(x: @SType("S") DTensor): DTensor
    @SType("S: Shape") fun ln(x: @SType("S") DTensor): DTensor
    @SType("S: Shape") fun lgamma(x: @SType("S") DTensor): DTensor
    @SType("S: Shape") fun digamma(x: @SType("S") DTensor): DTensor
    @SType("S: Shape") fun polygamma(n: Int, x: @SType("S") DTensor): DTensor
    @SType("S: Shape") fun sqrt(x: @SType("S") DTensor): DTensor
    @SType("S: Shape") fun tanh(x: @SType("S") DTensor): DTensor

    fun meld(values: List<DTensor>, derivativeId: DerivativeID): DTensor
    fun split(x: DTensor, shapes: List<Shape>): List<DTensor>

    @SType("S1: Shape, S2: Shape, A: Dim")
    fun concat(
        left: @SType("S1")  DTensor,
        right: @SType("S2") DTensor,
        axis: @SType("A") Int,
        derivativeId: DerivativeID
    ): @SType("concatOnAxis(S1, S2, A)") DTensor

    fun concat(slices: List<DTensor>, axis: Int, derivativeId: DerivativeID): DTensor

    @SType("S: Shape")
    fun broadcastTo(x: DTensor, newShape: @SType("S") Shape): @SType("S") DTensor
    /**
     * Applies convolution to the tensor signal, using filter. Both signal and filter must be of rank 4.
     * The expected shape of signal is NHWC (num signal, height, width, channels)
     * and the expected filter shape is OHWI (output channels, height, width, input channels) where C == I
     *
     * The type of padding to be applied to the signal can be of type
     * [PaddingStyle.Valid], [PaddingStyle.Same], [PaddingStyle.Full] or [PaddingStyle.Explicit]
     */
    fun convImpl(
        signal: DTensor,
        filter: DTensor,
        hStride: Int,
        vStride: Int,
        padding: Convolve.Padding2D,
        derivativeId: DerivativeID
    ): DTensor

    @SType("S: Shape")
    fun expand(x: DTensor, newShape: @SType("S") Shape): @SType("S") DTensor

    @SType("S: Shape")
    fun flip(x: @SType("S") DTensor, axes: IntArray): @SType("S") DTensor

    fun logSoftmax(x: DTensor, axis: Int): DTensor = baseLogSoftmax(x, axis)
    fun logSoftmaxGrad(x: DTensor, axis: Int, logSoftmax: DTensor, upstream: DTensor): DTensor

    @SType("S: Shape")
    fun pow(base: @SType("S") DTensor, exponent: Float): @SType("S") DTensor

    fun view1(x: DTensor, indices: IntArray): DTensor
    fun view2(x: DTensor, index: Int, axis: Int): DTensor
    fun view3(x: DTensor, index: IntRange, axis: Int): DTensor
    fun reshape(x: DTensor, newShape: Shape): DTensor
    fun reshapeToScalar(x: DTensor): DScalar
    fun squeeze(x: DTensor, axis: Int): DTensor
    fun unsqueeze(x: DTensor, axis: Int): DTensor
    fun transpose(x: DTensor, axes: IntArray): DTensor
    @SType("S: Shape") fun relu(x: @SType("S") DTensor): @SType("S") DTensor
    @SType("S: Shape") fun sigmoid(x: @SType("S") DTensor): @SType("S")DTensor
    fun reluGrad(x: DTensor, reluUpstream: DTensor, derivativeId: DerivativeID): DTensor
    fun sum(x: DTensor, axes: IntArray, keepDims: Boolean): DTensor
    fun avgPool(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor
    fun avgPoolGrad(x: DTensor, poolHeight: Int, poolWidth: Int): DTensor
    fun batchNorm(input: DTensor, scaleShift: DTensor, derivativeId: DerivativeID): BatchNormResult =
        baseBatchNorm(input, scaleShift)
    fun maxPoolWithIndices(
        x: DTensor,
        poolHeight: Int,
        poolWidth: Int,
        withIndices: Boolean): Pair<DTensor, List<IntArray>?>
    fun gather(x: DTensor, indices: List<Int>, axis: Int, paddingIndex: Int): DTensor
    fun gatherAtIndices(x: DTensor, indices: List<IntArray>): DTensor
    fun scatter(x: DTensor, indices: List<Int>, axis: Int, newShape: Shape, paddingIndex: Int): DTensor
    fun scatterAtIndices(x: DTensor, indices: List<IntArray>, newShape: Shape): DTensor
    fun gamma(alpha: DTensor, randomKey: RandomKey): DTensor

    @SType("S: Shape")
    fun compare(left: @SType("S") DTensor, right: @SType("S") DTensor, comparison: ComparisonKind): @SType("S") DTensor


    @SType("S: Shape")
    fun ifThenElse(condition: @SType("S") DTensor, whenTrue: @SType("S") DTensor, whenFalse: @SType("S") DTensor, derivativeId: DerivativeID): @SType("S") DTensor
}

/**
 * For a binary operation, find the common operations kind.
 */
internal fun commonKind(l: DTensor, r: DTensor): Pair<Operations, DerivativeID> {
    val ls = l.derivativeID.sequence
    val rs = r.derivativeID.sequence
    if (ls > rs) {
        val operations = if (l.isScalar && !r.isScalar) l.operations.tensor else l.operations
        return Pair(operations, l.derivativeID)
    }

    if (rs > ls || rs != 0) {
        val operations = if (r.isScalar && !l.isScalar) r.operations.tensor else r.operations
        return Pair(operations, r.derivativeID)
    }

    assert(rs == 0 && ls == 0)

    if (l.operations == TracingTensorOperations || r.operations == TracingTensorOperations)
        return Pair(TracingTensorOperations, NoDerivativeID)
    if (l.operations == SparseFloatTensorOperations || r.operations == SparseFloatTensorOperations)
        return Pair(SparseFloatTensorOperations, NoDerivativeID)
    if (l.operations == SparseRowFloatTensorOperations || r.operations == SparseRowFloatTensorOperations)
        return Pair(SparseRowFloatTensorOperations, NoDerivativeID)
    if (l.operations == StridedFloatTensorOperations || r.operations == StridedFloatTensorOperations)
        return Pair(StridedFloatTensorOperations, NoDerivativeID)
    if (l.operations == FloatScalarOperations && r.isScalar || r.operations == FloatScalarOperations && l.isScalar)
        return Pair(FloatScalarOperations, NoDerivativeID)
    if (l.operations == r.operations) // Sparse or GPU
        return Pair(l.operations, NoDerivativeID)
    throw Error("Incompatible tensor kinds: ${l.operations.name} and ${r.operations.name}")
}
