/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.reverse.ReverseDerivativeID
import org.diffkt.reverse.ReverseScalar
import org.diffkt.reverse.ReverseTensor

// ********** General Reverse Derivative of Univariate Functions **********

/**
 * Reverse Derivative for a function from [DTensor] to [DTensor].
 */
fun primalAndReverseDerivative(x: DTensor, f: (DTensor) -> DTensor): Pair<DTensor, DTensor> {
    return primalAndReverseDerivativeImpl(x, false, f)
}

// Primal and vector-Jacobian product
fun primalAndVjp(
    x: DTensor,
    v: DTensor,
    f: (x: DTensor) -> DTensor,
): Pair<DTensor, DTensor> = primalAndVjp(x, { v }, f)

// Primal and vector-Jacobian product
fun primalAndVjp(
    x: DTensor,
    vf: (primal: DTensor) -> DTensor,
    f: (x: DTensor) -> DTensor,
): Pair<DTensor, DTensor> {
    val (primal, pullback) = primalAndPullback(x, f)
    val v = vf(primal)
    require(v.shape == primal.shape) { "The shape of v was ${v.shape} but should be ${primal.shape}."}
    val vjp = pullback(v)
    return Pair(primal, vjp)
}

// vector-Jacobian product
fun vjp(
    x: DTensor,
    v: DTensor,
    f: (x: DTensor) -> DTensor,
) = primalAndVjp(x, v, f).second

fun vjp(
    x: DTensor,
    vf: (primal: DTensor) -> DTensor,
    f: (x: DTensor) -> DTensor,
) = primalAndVjp(x, vf, f).second

// Also known as VJP (vector-Jacobian product)
internal fun primalAndPullback(
    x: DTensor,
    f: (x: DTensor) -> DTensor
): Pair<DTensor, (DTensor) -> DTensor> {
    // Create a fresh epsilon
    val derivativeID = ReverseDerivativeID()

    // Make the inputs active variables
    val reverseX = when (x) {
        is DScalar -> ActiveVariableReverseScalar(x, derivativeID)
        else -> ActiveVariableReverseTensor(x, derivativeID)
    }

    // Compute the function
    val result = f(reverseX)
    var pulledBack = false

    val pullback: (DTensor) -> DTensor = { upstream: DTensor ->
        // The pullback is a one-shot function (sorry). Ensure it is called at most once.
        // Consider: Can we relax this restriction by keeping the upstreams in a side table?
        // Or keep it in a side table during non-initial invocations?  Or invocations on
        // another thread?
        if (pulledBack) throw IllegalStateException("pullback may only be invoked once")
        pulledBack = true

        // Set the upstream gradient shape
        derivativeID.upstreamShape = upstream.shape.drop(result.shape.rank)

        // Find the part of the primal result for this derivative
        var primalResult0 = result
        while (primalResult0.derivativeID.sequence > derivativeID.sequence)
            primalResult0 = primalResult0.primal

        if (primalResult0.derivativeID == derivativeID) {
            // Backpropagate the derivatives
            val initialUpstream = upstream
            primalResult0 as ReverseTensor
            primalResult0.pushback(initialUpstream)
        }

        val map = derivativeID.reversePass()

        // Extract the derivative from the active input
        map.get(reverseX)!!
    }

    val primal = result.primal(derivativeID)

    // Return the primal and pullback
    return Pair(primal, pullback)
}

private fun primalAndReverseDerivativeImpl(
    x: DTensor,
    scalarGradient: Boolean,
    f: (DTensor) -> DTensor
): Pair<DTensor, DTensor> {
    val (primalResult, pullback) = primalAndPullback(x, f)

    val initialUpstream = if (scalarGradient)
            FloatTensor.ones(primalResult.shape) // TODO: should delegate to an Operations
        else
            primalResult.operations.identityGradientOfSameKind(primalResult, primalResult.shape)
    val gradientResult = pullback(initialUpstream)
    return Pair(primalResult, gradientResult)
}

fun primalAndReverseDerivativeTransposed(x: DTensor, f: (DTensor) -> DTensor): Pair<DTensor, DTensor> {
    val pad = primalAndReverseDerivative(x, f)
    return Pair(pad.first, pad.second.leftTranspose(x.shape, pad.first.shape))
}

fun reverseDerivative(x: DTensor, f: (DTensor) -> DTensor): DTensor = primalAndReverseDerivative(x, f).second
fun reverseDerivative1(x: DTensor, f: (DTensor) -> DTensor): DTensor = reverseDerivative(x, f)
fun reverseDerivative2(x: DTensor, f: (DTensor) -> DTensor): DTensor = reverseDerivative(2, x, f)
fun reverseDerivative3(x: DTensor, f: (DTensor) -> DTensor): DTensor = reverseDerivative(3, x, f)
fun reverseDerivative4(x: DTensor, f: (DTensor) -> DTensor): DTensor = reverseDerivative(4, x, f)

// a function for computing the nth derivative (the 0th being the primal value)
fun reverseDerivative(n: Int, x: DTensor, f: (DTensor) -> DTensor): DTensor =
    if (n == 0) f(x) else reverseDerivative(x) { y: DTensor -> reverseDerivative(n-1, y, f) }

/**
 * Reverse gradient for a function from [DTensor] to [DTensor].
 */
fun primalAndGradient(x: DTensor, f: (DTensor) -> DScalar): Pair<DTensor, DTensor> =
    primalAndReverseDerivativeImpl(x, true, f)

internal class ActiveVariableReverseTensor(primal: DTensor, derivativeID: ReverseDerivativeID) : ReverseTensor(primal, derivativeID) {
    override fun backpropagate() {
        assert(upstream.derivativeID.sequence < derivativeID.sequence)
    }
}

internal class ActiveVariableReverseScalar(primal: DScalar, derivativeID: ReverseDerivativeID) : ReverseScalar(primal, derivativeID) {
    init {
        assert(derivativeID.sequence > primal.derivativeID.sequence)
    }
    override fun backpropagate() {
        assert(upstream.derivativeID.sequence < derivativeID.sequence)
    }
}

// ********** Reverse Derivative of Univariate Scalar Functions **********

/**
 * Reverse derivative of a function from [DScalar] to [DScalar].
 */
fun primalAndReverseDerivative(x: DScalar, f: (DScalar) -> DScalar): Pair<DScalar, DScalar> {
    val pad = primalAndReverseDerivative(x as DTensor) { f(it as DScalar) }
    return Pair(pad.first as DScalar, pad.second as DScalar)
}

fun reverseDerivative(x: DScalar, f: (DScalar) -> DScalar): DScalar = primalAndReverseDerivative(x, f).second

// differentiation as a high-order function
fun reverseDiff(f: (DScalar) -> DScalar): (DScalar) -> DScalar =
        { x: DScalar -> primalAndReverseDerivative(x, f).second }

// a function for computing the nth derivative (the 0th being the primal value)
fun reverseDerivative(n: Int, x: DScalar, f: (DScalar) -> DScalar): DScalar =
        if (n == 0) f(x) else reverseDerivative(x) { xx: DScalar -> reverseDerivative(n-1, xx, f) }
fun reverseDerivative1(x: DScalar, f: (DScalar) -> DScalar): DScalar = reverseDerivative(x) { xx: DScalar -> f(xx) }
fun reverseDerivative2(x: DScalar, f: (DScalar) -> DScalar): DScalar = reverseDerivative(2, x, f)
fun reverseDerivative3(x: DScalar, f: (DScalar) -> DScalar): DScalar = reverseDerivative(3, x, f)
fun reverseDerivative4(x: DScalar, f: (DScalar) -> DScalar): DScalar = reverseDerivative(4, x, f)

// ********** Reverse Derivative of Multivariate Functions **********

/**
 * The Reverse derivative of a multivariate function.
 */
fun primalAndReverseDerivative(inputs: List<DTensor>, f: (List<DTensor>) -> DTensor) : Pair<DTensor, List<DTensor>> {
    return primalAndReverseDerivativeImpl(inputs, false, f)
}

fun primalAndReverseDerivativeImpl(inputs: List<DTensor>, scalarGradient: Boolean, f: (List<DTensor>) -> DTensor) : Pair<DTensor, List<DTensor>> {
    val input = meld(inputs)
    val pad = primalAndReverseDerivativeImpl(input, scalarGradient) { wrappedInput: DTensor
        ->
        val inputShapes = inputs.map { it.shape }
        val splitInputs = wrappedInput.split(inputShapes)
        f(splitInputs)
    }
    val primalResult = pad.first
    val primalShape = if (scalarGradient) Shape() else primalResult.shape
    // for inputs of shape T<A> and T<B>, and result of shape T<R>,...
    val rawDerivative = pad.second // shape T<A+B,R>
    val derivativeResult = rawDerivative.split(inputs.map { it.shape + primalShape }) // T<A,R> and T<B,R>
    return Pair(primalResult, derivativeResult)
}

@JvmName("primalAndReverseDerivative_2")
fun primalAndReverseDerivative(inputs: List<DScalar>, f: (List<DScalar>) -> DScalar) : Pair<DScalar, List<DScalar>> {
    val (p, d) = primalAndReverseDerivative(inputs as List<DTensor>) { l -> f(l.map { it as DScalar }) }
    return Pair(p as DScalar, d.map { it as DScalar })
}

fun reverseDerivative(inputs: List<DScalar>, f: (List<DScalar>) -> DScalar) : List<DScalar> {
    val (_, d) = primalAndReverseDerivative(inputs as List<DTensor>) { l -> f(l.map { it as DScalar }) }
    return d.map { it as DScalar }
}

fun primalAndReverseDerivativeTransposed(inputs: List<DTensor>, f: (List<DTensor>) -> DTensor) : Pair<DTensor, List<DTensor>> {
    val pads = primalAndReverseDerivative(inputs, f)
    val primal = pads.first
    val derivativeTransposed = List(inputs.size) {
        pads.second[it].leftTranspose(inputs[it].shape, primal.shape)
    }
    return Pair(primal, derivativeTransposed)
}

/**
 * The Reverse gradients of a multivariate function.
 */
@Suppress("UNCHECKED_CAST")
fun primalAndGradient(inputs: List<DTensor>, f: (List<DTensor>) -> DScalar) : Pair<DScalar, List<DTensor>> =
    primalAndReverseDerivativeImpl(inputs, true, f) as Pair<DScalar, List<DTensor>>

/**
 * The Reverse derivative of a multivariate tensor function.
 */
fun primalAndReverseDerivative(x: DTensor, y: DTensor, f: (DTensor, DTensor) -> DTensor): Pair<DTensor, Pair<DTensor, DTensor>> {
    val pad = primalAndReverseDerivative(listOf(x, y)) { args: List<DTensor> ->
        assert(args.size == 2)
        f(args[0], args[1])
    }
    assert(pad.second.size == 2)
    return Pair(pad.first, Pair(pad.second[0], pad.second[1]))
}

@Suppress("UNCHECKED_CAST")
fun primalAndGradient(x: DTensor, y: DTensor, f: (DTensor, DTensor) -> DScalar): Pair<DScalar, Pair<DTensor, DTensor>> =
    (primalAndReverseDerivativeImpl(listOf(x,y), true) { args: List<DTensor> ->
        assert(args.size == 2)
        f(args[0], args[1])
    }).let { Pair(it.first, Pair(it.second[0], it.second[1]))} as Pair<DScalar, Pair<DTensor, DTensor>>

fun reverseDerivative(x: DTensor, y: DTensor, f: (DTensor, DTensor) -> DTensor): Pair<DTensor, DTensor> =
    primalAndReverseDerivative(x, y, f).second

fun primalAndReverseDerivativeTransposed(x: DTensor, y: DTensor, f: (DTensor, DTensor) -> DTensor): Pair<DTensor, Pair<DTensor, DTensor>> {
    val pad = primalAndReverseDerivativeTransposed(listOf(x, y), { xx: List<DTensor> -> f(xx[0], xx[1]) })
    return Pair(pad.first, Pair(pad.second[0], pad.second[1]))
}

fun reverseDerivativeTransposed(x: DTensor, y: DTensor, f: (DTensor, DTensor) -> DTensor): Pair<DTensor, DTensor> =
    primalAndReverseDerivativeTransposed(x, y, f).second

/**
 * The Reverse derivative of a multivariate scalar function.
 */
fun primalAndReverseDerivative(x: DScalar, y: DScalar, f: (DScalar, DScalar) -> DScalar): Pair<DScalar, Pair<DScalar, DScalar>> {
    val pad = primalAndReverseDerivative(listOf(x, y)) { args: List<DScalar> ->
        assert(args.size == 2)
        f(args[0], args[1])
    }
    assert(pad.second.size == 2)
    return Pair(pad.first, Pair(pad.second[0], pad.second[1]))
}

fun reverseDerivative(x: DScalar, y: DScalar, f: (DScalar, DScalar) -> DScalar): Pair<DScalar, DScalar> =
        primalAndReverseDerivative(x, y, f).second

fun primalAndReverseDerivativeTransposed(x: DScalar, y: DScalar, f: (DScalar, DScalar) -> DScalar): Pair<DScalar, Pair<DScalar, DScalar>> {
    val pad =
        primalAndReverseDerivativeTransposed(listOf(x, y), { xx: List<DTensor> -> f(xx[0] as DScalar, xx[1] as DScalar) })
    return Pair(pad.first as DScalar, Pair(pad.second[0] as DScalar, pad.second[1] as DScalar))
}

fun reverseDerivativeTransposed(x: DScalar, y: DScalar, f: (DScalar, DScalar) -> DScalar): Pair<DScalar, DScalar> =
    primalAndReverseDerivativeTransposed(x, y, f).second
