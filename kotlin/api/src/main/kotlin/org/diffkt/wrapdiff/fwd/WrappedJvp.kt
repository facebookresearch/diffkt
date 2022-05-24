/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.forward.ForwardDerivativeID
import org.diffkt.forward.ForwardTensor

fun <Input : Any, Output : Any, InputTangent : Any, OutputTangent : Any> primalAndJvp(
    x: Input, v: InputTangent,
    f: (Input) -> Output,
    makeForwardInput: (primal: Input, tangent: InputTangent, makeForwardTensor: (primal: DTensor, tangent: DTensor) -> DTensor) -> Input,
    wrapOutput: ((Output, Wrapper) -> Output)? = null,
    extractTangent: (Output, extractTensorTangent: (outputTensor: DTensor) -> DTensor) -> OutputTangent,
): Pair<Output, OutputTangent> {
    // Wrap the input
    val derivativeID = ForwardDerivativeID(Shape())
    fun makeForwardTensor(primal: DTensor, tangent: DTensor): DTensor {
        return ForwardTensor(primal, derivativeID, tangent)
    }
    val wrappedInput = makeForwardInput(x, v, ::makeForwardTensor)

    // Compute the function
    val wrappedOutput = f(wrappedInput)

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

    fun extractTensorTangent(outputTensor: DTensor): DTensor {
        return if (outputTensor is ForwardTensor && outputTensor.derivativeID == derivativeID)
            outputTensor.tangent
        else
            outputTensor.operations.zeroOfSameKind(outputTensor, outputTensor.shape)
    }

    val outputTangent = extractTangent(wrappedOutput, ::extractTensorTangent)

    return Pair(unwrappedOutput, outputTangent)
}

fun <Input : Any, Output : Any, OutputTangent : Any> primalAndJvp(
    x: Input, v: Input,
    f: (Input) -> Output,
    wrapInput: ((Input, Wrapper) -> Input)? = null,
    wrapOutput: ((Output, Wrapper) -> Output)? = null,
    extractOutputTangent: (Output, extractTensorTangent: (outputTensor: DTensor) -> DTensor) -> OutputTangent,
): Pair<Output, OutputTangent> {
    return primalAndJvp(
        x, v, f,
        defaultMakeForwardInput(wrapInput),
        wrapOutput,
        extractOutputTangent
    )
}

fun <Input : Any, Output : Any> primalAndJvp(
    x: Input, v: Input,
    f: (Input) -> Output,
    wrapInput: ((Input, Wrapper) -> Input)? = null,
    wrapOutput: ((Output, Wrapper) -> Output)? = null,
): Pair<Output, Output> {
    return primalAndJvp(
        x, v, f,
        wrapInput = wrapInput,
        wrapOutput = wrapOutput,
        extractOutputTangent = defaultExtractOutputTangent(wrapOutput)
    )
}