/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.forward.ForwardDerivativeID
import org.diffkt.forward.ForwardScalar
import org.diffkt.forward.ForwardTensor

// ********** General Forward Derivative of Univariate Functions **********

/**
 * Forward derivative of a function from [DTensor] to [DTensor].
 */
fun primalAndForwardDerivative(x: DTensor, f: (DTensor) -> DTensor): Pair<DTensor, DTensor> {
    // Create a fresh epsilon
    val derivativeID = ForwardDerivativeID(x.shape)

    // Join that epsilon value (times one) to x.
    val xpe = ForwardTensor(x, derivativeID, x.operations.identityGradientOfSameKind(x, x.shape))

    // Compute the function
    val result = f(xpe)

    // Extract the primal and derivative
    var v = result
    while (v.derivativeID.sequence > derivativeID.sequence)
        v = v.primal
    if (v.derivativeID == derivativeID) {
        val tangent = when (v) {
            is ForwardScalar -> v.tangent
            is ForwardTensor -> v.tangent
            else -> throw Error()
        }
        return Pair(v.primal, tangent)
    }

    return Pair(v, zeroOfSameKind(v, v.shape + derivativeID.inputTangentShapeForJacobian))
}

// Primal and Jacobian-vector product
fun primalAndJvp(x: DTensor, v: DTensor, f: (DTensor) -> DTensor): Pair<DTensor, DTensor> {
    require(v.shape == x.shape) { "The shape of v was ${v.shape} but should be ${x.shape}."}

    // Create a fresh epsilon
    val derivativeID = ForwardDerivativeID(Shape())

    // Join the input and the initial gradient
    val xpe = ForwardTensor(x, derivativeID, v)

    // Compute the function
    var result = f(xpe)

    // Extract the primal and derivative
    while (result.derivativeID.sequence > derivativeID.sequence)
        result = result.primal
    if (result.derivativeID == derivativeID) {
        result as ForwardTensor
        val jvp = result.tangent
        return Pair(result.primal, jvp)
    }

    return Pair(result, zeroOfSameKind(result, result.shape + derivativeID.inputTangentShapeForJacobian))
}

// Jacobian-vector product
fun jvp(x: DTensor, v: DTensor, f: (DTensor) -> DTensor) = primalAndJvp(x, v, f).second

fun forwardDerivative(x: DTensor, f: (DTensor) -> DTensor): DTensor = primalAndForwardDerivative(x, f).second

fun forwardDerivative1(x: DTensor, f: (DTensor) -> DTensor): DTensor = forwardDerivative(x, f)
fun forwardDerivative2(x: DTensor, f: (DTensor) -> DTensor): DTensor = forwardDerivative(2, x, f)
fun forwardDerivative3(x: DTensor, f: (DTensor) -> DTensor): DTensor = forwardDerivative(3, x, f)
fun forwardDerivative4(x: DTensor, f: (DTensor) -> DTensor): DTensor = forwardDerivative(4, x, f)

// a function for computing the nth derivative (the 0th being the primal value)
fun forwardDerivative(n: Int, x: DTensor, f: (DTensor) -> DTensor): DTensor = if (n == 0) f(x) else forwardDerivative(x) { y: DTensor -> forwardDerivative(n-1, y, f) }

// ********** Forward Derivative of Univariate Scalar Functions **********

/**
 * Forward derivative of a function from [DScalar] to [DScalar].
 */
fun primalAndForwardDerivative(x: DScalar, f: (DScalar) -> DScalar): Pair<DScalar, DScalar> {
    val pad = primalAndForwardDerivative(x as DTensor) { xx: DTensor -> f(xx as DScalar) }
    return Pair(pad.first as DScalar, pad.second as DScalar)
}

fun forwardDerivative(x: DScalar, f: (DScalar) -> DScalar): DScalar = primalAndForwardDerivative(x, f).second

// differentiation as a high-order function
fun forwardDiff(f: (DScalar) -> DScalar): (DScalar) -> DScalar =
    { x: DScalar -> primalAndForwardDerivative(x, f).second }

// a function for computing the nth derivative (the 0th being the primal value)
fun forwardDerivative(n: Int, x: DScalar, f: (DScalar) -> DScalar): DScalar =
        if (n == 0) f(x) else forwardDerivative(x) { xx: DScalar -> forwardDerivative(n-1, xx, f) }
fun forwardDerivative1(x: DScalar, f: (DScalar) -> DScalar): DScalar = forwardDerivative(x) { xx: DScalar -> f(xx) }
fun forwardDerivative2(x: DScalar, f: (DScalar) -> DScalar): DScalar = forwardDerivative(2, x, f)
fun forwardDerivative3(x: DScalar, f: (DScalar) -> DScalar): DScalar = forwardDerivative(3, x, f)
fun forwardDerivative4(x: DScalar, f: (DScalar) -> DScalar): DScalar = forwardDerivative(4, x, f)

// ********** Forward Derivative of Multivariate Functions **********

/**
 * The forward derivative of a multivariate function.
 */
fun primalAndForwardDerivative(inputs: List<DTensor>, f: (List<DTensor>) -> DTensor) : Pair<DTensor, List<DTensor>> {
    val input: DTensor = meld(inputs)
    val pad = primalAndForwardDerivative(input) { wrappedInput: DTensor
        ->
        val inputShapes = inputs.map { it.shape }
        val splitInputs = wrappedInput.split(inputShapes)
        f(splitInputs)
    }
    val primalResult = pad.first
    val primalShape = primalResult.shape
    // For inputs of shapes T<A> and T<B>, and output of shape T<R>, ...
    val derivativeResult = pad.second // shape T<R,A+B>
    val derivativeTransposed = derivativeResult.leftTranspose(primalShape, input.shape) // shape T<A+B,R>
    val splitTransposedResult = derivativeTransposed.split(
            inputs.map { it.shape + primalShape }) // shape T<A,R> and T<B,R>
    val finalderivative = splitTransposedResult.map {
        it.leftTranspose(it.shape.dropLast(primalResult.rank), primalShape) } // shape T<R,A> and T<R,B>
    return Pair(primalResult, finalderivative)
}

/**
 * The forward derivative of a multivariate tensor function.
 */
fun primalAndForwardDerivative(x: DTensor, y: DTensor, f: (DTensor, DTensor) -> DTensor): Pair<DTensor, Pair<DTensor, DTensor>> {
    val pad = primalAndForwardDerivative(listOf(x, y)) { args: List<DTensor> ->
        assert(args.size == 2)
        f(args[0], args[1])
    }
    assert(pad.second.size == 2)
    return Pair(pad.first, Pair(pad.second[0], pad.second[1]))
}

@JvmName("primalAndForwardderivative_2")
fun primalAndForwardDerivative(inputs: List<DScalar>, f: (List<DScalar>) -> DScalar) : Pair<DScalar, List<DScalar>> {
    val (p, d) = primalAndForwardDerivative(inputs as List<DTensor>) { l -> f(l.map { it as DScalar }) }
    return Pair(p as DScalar, d.map { it as DScalar })
}


fun forwardDerivative(x: DTensor, y: DTensor, f: (DTensor, DTensor) -> DTensor): Pair<DTensor, DTensor> =
    primalAndForwardDerivative(x, y, f).second

/**
 * The forward derivative of a multivariate scalar function.
 */
fun primalAndForwardDerivative(x: DScalar, y: DScalar, f: (DScalar, DScalar) -> DScalar): Pair<DScalar, Pair<DScalar, DScalar>> {
    val pad = primalAndForwardDerivative(listOf(x, y)) { args: List<DScalar> ->
        assert(args.size == 2)
        f(args[0], args[1])
    }
    assert(pad.second.size == 2)
    return Pair(pad.first, Pair(pad.second[0], pad.second[1]))
}

fun forwardDerivative(x: DScalar, y: DScalar, f: (DScalar, DScalar) -> DScalar): Pair<DScalar, DScalar> =
    primalAndForwardDerivative(x, y, f).second
