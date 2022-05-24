/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.reverse.ReverseDerivativeID
import org.diffkt.reverse.ReverseTensor

internal fun <Input : Any, Output : Any, InputTangent : Any, OutputTangent : Any> primalAndPullback(
    x: Input,
    f: (Input) -> Output,
    wrapOutput: ((Output, Wrapper) -> Output)? = null,
    makeReverseInput: (primal: Input, makeReverseTensor: (primal: DTensor) -> DTensor) -> Input,
    setOutputTangent: (Output, OutputTangent, setTensorTangent: (tensor: DTensor, tangent: DTensor) -> Unit) -> Unit,
    extractInputTangent: (Input, extractTensorTangent: (DTensor) -> DTensor) -> InputTangent
): Pair<Output, (OutputTangent) -> InputTangent> {
    // Wrap the input
    val derivativeID = ReverseDerivativeID()
    fun makeReverseTensor(primal: DTensor): DTensor {
        return when (primal) {
            is DScalar -> ActiveVariableReverseScalar(primal, derivativeID)
            else -> ActiveVariableReverseTensor(primal, derivativeID)
        }
    }

    val wrappedInput = makeReverseInput(x, ::makeReverseTensor)

    // Compute the function
    val wrappedOutput = f(wrappedInput)

    fun setTensorTangent(tensor: DTensor, tangent: DTensor): Unit {
        if (tensor is FloatTensor) return
        require(tensor is ReverseTensor) {
            "expected ReverseTensor, found ${tensor}"
        }
        tensor.pushback(tangent)
    }

    // Create the pullback
    var pulledBack = false
    fun pullback(db: OutputTangent): InputTangent {
        if (pulledBack) throw IllegalStateException("pullback may only be invoked once")
        pulledBack = true

        derivativeID.upstreamShape = Shape()
        setOutputTangent(wrappedOutput, db, ::setTensorTangent)
        val map = derivativeID.reversePass()
        fun extractTensorTangent(tensor: DTensor): DTensor = map[tensor]!!
        return extractInputTangent(wrappedInput, ::extractTensorTangent)
    }

    // Unwrap the output
    val outputUnwrapper = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            var primalResult = value
            while (primalResult.derivativeID.sequence > derivativeID.sequence) primalResult = primalResult.primal
            if (primalResult.derivativeID == derivativeID) primalResult = primalResult.primal
            return primalResult
        }
    }

    val unwrappedOutput = if (wrapOutput != null)
        wrapOutput(wrappedOutput, outputUnwrapper)
    else
        outputUnwrapper.wrap(wrappedOutput)

    return Pair(unwrappedOutput, ::pullback)
}

internal fun <Input : Any, Output : Any, InputTangent : Any> primalAndPullback(
    x: Input,
    f: (Input) -> Output,
    wrapOutput: ((Output, Wrapper) -> Output)? = null,
    makeReverseInput: (primal: Input, makeReverseTensor: (primal: DTensor) -> DTensor) -> Input,
    extractInputTangent: (Input, extractTensorTangent: (DTensor) -> DTensor) -> InputTangent,
): Pair<Output, (Output) -> InputTangent> {
    return primalAndPullback(
        x, f,
        wrapOutput = wrapOutput,
        makeReverseInput = makeReverseInput,
        setOutputTangent = defaultSetOutputTangent(wrapOutput),
        extractInputTangent = extractInputTangent
    )
}

internal fun <Input : Any, Output : Any, OutputTangent : Any> primalAndPullback(
    x: Input,
    f: (Input) -> Output,
    wrapInput: ((Input, Wrapper) -> Input)? = null,
    wrapOutput: ((Output, Wrapper) -> Output)? = null,
    setOutputTangent: (Output, OutputTangent, setTensorTangent: (tensor: DTensor, tangent: DTensor) -> Unit) -> Unit
): Pair<Output, (OutputTangent) -> Input> {
    return primalAndPullback(
        x, f,
        wrapOutput = wrapOutput,
        makeReverseInput = defaultMakeReverseInput(wrapInput),
        setOutputTangent = setOutputTangent,
        extractInputTangent = defaultExtractInputTangent(wrapInput)
    )
}

internal fun <Input : Any, Output : Any> primalAndPullback(
    x: Input,
    f: (Input) -> Output,
    wrapInput: ((Input, Wrapper) -> Input)? = null,
    wrapOutput: ((Output, Wrapper) -> Output)? = null,
): Pair<Output, (Output) -> Input> {
    val setOutputTangent = defaultSetOutputTangent(wrapOutput)
    return primalAndPullback(
        x, f,
        wrapOutput = wrapOutput,
        makeReverseInput = defaultMakeReverseInput(wrapInput),
        setOutputTangent = setOutputTangent,
        extractInputTangent = defaultExtractInputTangent(wrapInput)
    )
}