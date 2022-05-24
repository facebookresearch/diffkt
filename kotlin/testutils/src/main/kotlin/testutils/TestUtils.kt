/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package testutils

import io.kotest.assertions.collectOrThrow
import io.kotest.assertions.errorCollector
import org.diffkt.*
import io.kotest.matchers.floats.*
import org.diffkt.SparseFloatTensor
import org.junit.jupiter.api.Assertions.assertTrue
import kotlin.math.*

val DScalar.value: Float get() {
    return when (val sc = this.primal(NoDerivativeID)) {
        is FloatScalar -> sc.value
        else -> throw IllegalArgumentException("cannot get the value from a ${sc::class.qualifiedName}")
    }
}

fun floats(n: Int, start: Int = 1, repeat: Int = 1): FloatArray =
    FloatArray(n * repeat) { (it % n + start).toFloat() }

fun isClose(x: Float, y: Float) = (x == y)
        || (x - y).absoluteValue / (x.absoluteValue + y.absoluteValue) < 0.00002
        || (x - y).absoluteValue < 0.00002

// TODO: replace assertTrue with shouldBe
fun assertClose(expected: Float, actual: Float) =
        assertTrue(isClose(expected, actual), "Expected: $expected, Actual: $actual")

infix fun Shape.shouldBe(expected: Shape) {
    if (!(this.dims contentEquals expected.dims))
        errorCollector.collectOrThrow(AssertionError("actual $this expected $expected"))
}

infix fun DTensor.shouldBeExactly(expected: DTensor) {
    this.shape shouldBe expected.shape
    val self = this.basePrimal()
    val exp = expected.basePrimal()
    val badPos = (0 until this.size).filter { pos ->
        val x = self.at(pos)
        val y = exp.at(pos)
        if (!x.isNaN() || !y.isNaN() && (!x.isInfinite() || !y.isInfinite()))
            x != y
        else
            false
    }
    if (badPos.isEmpty())
        return
    val pos = badPos[0]
    val x = self.at(pos)
    val y = exp.at(pos)
    val indexString = self.posToIndex(pos).joinToString()
    val nOthers = badPos.size - 1
    errorCollector.collectOrThrow(AssertionError(
        "wrong value at index [$indexString]: found ${x}f but expected ${y}f (and $nOthers others)\nexpected:\n$exp\nactual:\n$self"))
}

fun DTensor.shouldBeNear(expected: Float, epsilon: Float) {
    this.shouldBeNear(FloatScalar(expected), epsilon)
}

fun DTensor.shouldBeNear(expected: DTensor, epsilon: Float) {
    this.shape shouldBe expected.shape
    val self = this.basePrimal()
    val exp = expected.basePrimal()
    for (i in 0 until this.size) {
        val x = self.at(i)
        val y = exp.at(i)
        if ((!x.isNaN() || !y.isNaN()) && (!x.isInfinite() || !y.isInfinite()))
            abs(x - y) shouldBeLessThan epsilon
    }
}

@JvmName("shouldBeExactly_1")
infix fun Pair<DTensor, DTensor>.shouldBeExactly(expected: Pair<DTensor, DTensor>) {
    this.first shouldBeExactly expected.first
    this.second shouldBeExactly expected.second
}

@JvmName("shouldBeExactly_2")
infix fun Pair<DTensor, Pair<DTensor, DTensor>>.shouldBeExactly(expected: Pair<DTensor, Pair<DTensor, DTensor>>) {
    this.first shouldBeExactly expected.first
    this.second shouldBeExactly expected.second
}

infix fun DScalar.shouldBeExactly(expected: DScalar) {
    val self = this.primal(NoDerivativeID) as FloatScalar
    val exp = expected.primal(NoDerivativeID) as FloatScalar
    self.value shouldBeExactly exp.value
}

/**
 * Given sparse [inputs], compare operation [f] with [inputs] across both
 * sparse and dense formats.
 * Compare the reverseDerivative of [f] with the given [inputs] across sparse
 * and dense formats
 */
fun compareSparseWithDense(inputs: Pair<SparseFloatTensor, SparseFloatTensor>,
                           f: (x: DTensor, y: DTensor) -> DTensor,
                           testF: Boolean=true,
                           testReverseF: Boolean=true) {

    if (!testF && !testReverseF)
        return

    val t1Dense = inputs.first.toDense()
    val t2Dense = inputs.second.toDense()

    if (testF) {
        val out = f(inputs.first, inputs.second)
        val outDense = f(t1Dense, t2Dense)
        assert(out is SparseFloatTensor)
        out.shouldBeNear(outDense, 1e-6f)
    }

    if (testReverseF) {
        // reverse derivative
        val outReverse = reverseDerivative(inputs.first) { x: DTensor -> f(x, inputs.second) }
        val outReverseDense = reverseDerivative(t1Dense) { x: DTensor -> f(x, t2Dense) }

        outReverse.shouldBeNear(outReverseDense, 1e-6f)
    }
}

/**
 * Given sparse [input], compare operation [f] with [input] across both
 * sparse and dense formats.
 * Compare the reverseDerivative of [f] with the given [input] across sparse
 * and dense formats
 */
fun compareSparseWithDense(input: SparseFloatTensor,
                           f: (x: DTensor) -> DTensor,
                           testF: Boolean=true,
                           testReverseF: Boolean=true) {

    if (!testF && !testReverseF)
        return

    val inputDense = input.toDense()

    if (testF) {
        val out = f(input)
        val outDense = f(inputDense)
        assert(out is SparseFloatTensor)
        out.shouldBeNear(outDense, 1e-6f)
    }

    if (testReverseF) {
        // reverse derivative
        val outReverse = reverseDerivative(input) { x: DTensor -> f(x) }
        val outReverseDense = reverseDerivative(inputDense) { x: DTensor -> f(x) }

        outReverse.shouldBeNear(outReverseDense, 1e-6f)
    }
}

fun Float.shouldBeNear(expected: Float, delta: Float) {
    (this - expected).absoluteValue shouldBeLessThanOrEqual delta
}

infix fun Float.shouldBeCloseTo(expected: Float) {
    val epsilon = 0.005f
    abs(this - expected) shouldBeLessThanOrEqual  abs(expected) * 0.05f + epsilon
}

infix fun Pair<Float, Float>.shouldBeCloseTo(expected: Pair<Float, Float>) {
    val epsilon = 0.005f
    abs(this.first - expected.first) shouldBeLessThanOrEqual  abs(expected.first) * 0.05f + epsilon
    abs(this.second - expected.second) shouldBeLessThanOrEqual  abs(expected.second) * 0.05f + epsilon
}

val DTensor.scalarValue: Float get() {
    return when (val t = this.primal(NoDerivativeID)) {
        is FloatScalar -> t.value
        else -> throw IllegalArgumentException("cannot get scalar value from a ${t::class.qualifiedName}")
    }
}

fun stats(sample: DTensor): Pair<Float, Float> {
    val mean = (sample.sum() / sample.size).value
    val variance = ((sample - mean).pow(2).sum() / sample.size).value
    return Pair(mean, variance)
}
