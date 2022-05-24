/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.tracing.TracingTensor
import org.diffkt.tracing.TracingTensorOperations

operator fun Float.compareTo(right: DScalar) = this.compareTo(right.basePrimal().value)

operator fun DScalar.compareTo(right: Float) = this.basePrimal().compareTo(right)

operator fun DScalar.compareTo(right: DScalar) = this.basePrimal().compareTo(right.basePrimal())

operator fun Float.compareTo(right: FloatScalar) = this.compareTo(right.value)
operator fun FloatScalar.compareTo(right: Float) = this.value.compareTo(right)
operator fun FloatScalar.compareTo(right: FloatScalar) = this.value.compareTo(right.value)

enum class ComparisonKind {
    LT,
    GT,
    LE,
    GE,
    EQ,
    NE
}
val ComparisonKind.inverted get(): ComparisonKind {
    return when (this) {
        ComparisonKind.LT -> ComparisonKind.GE
        ComparisonKind.GT -> ComparisonKind.LE
        ComparisonKind.EQ -> ComparisonKind.NE
        ComparisonKind.NE -> ComparisonKind.EQ
        ComparisonKind.GE -> ComparisonKind.LT
        ComparisonKind.LE -> ComparisonKind.GT
    }
}

internal fun compare(l: Float, r: Float, comparison: ComparisonKind): Boolean {
    return when (comparison) {
        ComparisonKind.NE -> l != r
        ComparisonKind.LT -> l < r
        ComparisonKind.LE -> l <= r
        ComparisonKind.GE -> l >= r
        ComparisonKind.EQ -> l == r
        ComparisonKind.GT -> l > r
    }
}

internal fun compare(left: DTensor, right: DTensor, comparison: ComparisonKind): DTensor {
    val (l, r) = Broadcasting.broadcastToCommonShape(
        left.primal(NoDerivativeID), right.primal(NoDerivativeID))
    if (l.operations == r.operations)
        return l.operations.compare(l, r, comparison)
    assert(l is TracingTensor || r is TracingTensor)
    return TracingTensorOperations.compare(l, r, comparison)
}

internal fun compare(left: DScalar, right: DScalar, comparison: ComparisonKind): DScalar {
    return compare(left as DTensor, right as DTensor, comparison) as DScalar
}

infix fun Float.gt(other: Float) = compare(this, other, ComparisonKind.GT)
infix fun Float.ge(other: Float) = compare(this, other, ComparisonKind.GE)
infix fun Float.lt(other: Float) = compare(this, other, ComparisonKind.LT)
infix fun Float.le(other: Float) = compare(this, other, ComparisonKind.LE)
infix fun Float.eq(other: Float) = compare(this, other, ComparisonKind.EQ)
infix fun Float.ne(other: Float) = compare(this, other, ComparisonKind.NE)

infix fun DTensor.gt(other: DTensor) = compare(this, other, ComparisonKind.GT)
infix fun DTensor.ge(other: DTensor) = compare(this, other, ComparisonKind.GE)
infix fun DTensor.lt(other: DTensor) = compare(this, other, ComparisonKind.LT)
infix fun DTensor.le(other: DTensor) = compare(this, other, ComparisonKind.LE)
infix fun DTensor.eq(other: DTensor) = compare(this, other, ComparisonKind.EQ)
infix fun DTensor.ne(other: DTensor) = compare(this, other, ComparisonKind.NE)

infix fun DTensor.gt(other: Float) = compare(this, FloatScalar(other), ComparisonKind.GT)
infix fun DTensor.ge(other: Float) = compare(this, FloatScalar(other), ComparisonKind.GE)
infix fun DTensor.lt(other: Float) = compare(this, FloatScalar(other), ComparisonKind.LT)
infix fun DTensor.le(other: Float) = compare(this, FloatScalar(other), ComparisonKind.LE)
infix fun DTensor.eq(other: Float) = compare(this, FloatScalar(other), ComparisonKind.EQ)
infix fun DTensor.ne(other: Float) = compare(this, FloatScalar(other), ComparisonKind.NE)

infix fun Float.gt(other: DTensor) = compare(FloatScalar(this), other, ComparisonKind.GT)
infix fun Float.ge(other: DTensor) = compare(FloatScalar(this), other, ComparisonKind.GE)
infix fun Float.lt(other: DTensor) = compare(FloatScalar(this), other, ComparisonKind.LT)
infix fun Float.le(other: DTensor) = compare(FloatScalar(this), other, ComparisonKind.LE)
infix fun Float.eq(other: DTensor) = compare(FloatScalar(this), other, ComparisonKind.EQ)
infix fun Float.ne(other: DTensor) = compare(FloatScalar(this), other, ComparisonKind.NE)

infix fun DScalar.gt(other: DScalar) = compare(this, other, ComparisonKind.GT)
infix fun DScalar.ge(other: DScalar) = compare(this, other, ComparisonKind.GE)
infix fun DScalar.lt(other: DScalar) = compare(this, other, ComparisonKind.LT)
infix fun DScalar.le(other: DScalar) = compare(this, other, ComparisonKind.LE)
infix fun DScalar.eq(other: DScalar) = compare(this, other, ComparisonKind.EQ)
infix fun DScalar.ne(other: DScalar) = compare(this, other, ComparisonKind.NE)

infix fun DScalar.gt(other: Float) = compare(this, FloatScalar(other), ComparisonKind.GT)
infix fun DScalar.ge(other: Float) = compare(this, FloatScalar(other), ComparisonKind.GE)
infix fun DScalar.lt(other: Float) = compare(this, FloatScalar(other), ComparisonKind.LT)
infix fun DScalar.le(other: Float) = compare(this, FloatScalar(other), ComparisonKind.LE)
infix fun DScalar.eq(other: Float) = compare(this, FloatScalar(other), ComparisonKind.EQ)
infix fun DScalar.ne(other: Float) = compare(this, FloatScalar(other), ComparisonKind.NE)

infix fun Float.gt(other: DScalar) = compare(FloatScalar(this), other, ComparisonKind.GT)
infix fun Float.ge(other: DScalar) = compare(FloatScalar(this), other, ComparisonKind.GE)
infix fun Float.lt(other: DScalar) = compare(FloatScalar(this), other, ComparisonKind.LT)
infix fun Float.le(other: DScalar) = compare(FloatScalar(this), other, ComparisonKind.LE)
infix fun Float.eq(other: DScalar) = compare(FloatScalar(this), other, ComparisonKind.EQ)
infix fun Float.ne(other: DScalar) = compare(FloatScalar(this), other, ComparisonKind.NE)
