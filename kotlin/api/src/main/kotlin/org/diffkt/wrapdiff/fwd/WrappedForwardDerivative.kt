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
import shapeTyping.annotations.SType

/**
 * Evaluate the function, and return a pair containing its result (primal) and the derivative.
 * This version supports user-defined types for input and output of the function.
 */
fun <Input : Any, Output : Any, Derivative : Any> primalAndForwardDerivative(
    x: Input,
    f: (Input) -> Output,
    wrapInput: ((Input, Wrapper) -> Input)? = null,
    wrapOutput: ((Output, Wrapper) -> Output)? = null,
    extractDerivative: (input: Input, output: Output, extractor: (input: DTensor, output: DTensor) -> DTensor) -> Derivative,
): Pair<Output, Derivative> {
    // Wrap the input
    val derivativeID = object : ForwardDerivativeID() {
        fun fixInputTangentShape(shape: Shape) {
            this.inputTangentShapeForJacobian = shape
        }
    }

    // Make the inputs active variables
    val gatheredInputs: SequentialIntegerAssigner<ForwardTensor> = SequentialIntegerAssigner()
    val inputWrapper = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            return when (value) {
                is DScalar -> {
                    val rs = object : ForwardScalar(value, derivativeID) {}
                    gatheredInputs.add(rs)
                    rs
                }
                else -> {
                    val rt = object : ForwardTensor(value, derivativeID) {}
                    gatheredInputs.add(rt)
                    rt
                }
            }
        }
    }
    val wrappedInput = if (wrapInput != null) wrapInput(x, inputWrapper) else inputWrapper.wrap(x)

    // Check if the primal and tangent can have the same shape
    val scalarTangent = gatheredInputs.size == 0 || gatheredInputs.size == 1 && gatheredInputs[0].isScalar

    // Compute the shape of the forward tangents.
    val totalInputSize = gatheredInputs.map { it.size }.sum()
    val tangentShape = if (scalarTangent) Shape() else Shape(totalInputSize)
    derivativeID.fixInputTangentShape(tangentShape)

    // Assign tangent fields to the wrapped inputs.
    if (gatheredInputs.size > 0) {
        val firstInput = gatheredInputs[0]
        val wholeTangent = firstInput.operations.identityGradientOfSameKind(firstInput, tangentShape)
        val splitTangent = if (scalarTangent) listOf(wholeTangent) else wholeTangent.split(
            gatheredInputs.map { it.shape + tangentShape }
        )
        for (i in 0 until gatheredInputs.size) {
            gatheredInputs[i].tangent = splitTangent[i]
        }
    }

    // Compute the function
    val wrappedOutput = f(wrappedInput)

    // Unwrap the output, gathering the output variables
    val gatheredOutputs: SequentialIntegerAssigner<DTensor> = SequentialIntegerAssigner()
    val outputUnwrapper = object : Wrapper() {
        private fun unwrapOutput(wrappedOutput: DTensor): DTensor {
            var primalResult = wrappedOutput
            while (primalResult.derivativeID.sequence > derivativeID.sequence)
                primalResult = primalResult.primal
            if (primalResult.derivativeID == derivativeID) {
                gatheredOutputs.add(primalResult)
                primalResult = primalResult.primal
            }
            return primalResult
        }

        override fun wrapDTensor(value: DTensor): DTensor {
            return when (value) {
                is DScalar -> unwrapOutput(value)
                else -> unwrapOutput(value)
            }
        }
    }
    val unwrappedOutput = if (wrapOutput != null)
        wrapOutput(wrappedOutput, outputUnwrapper)
    else
        outputUnwrapper.wrap(wrappedOutput)

    // Extract the derivative from the outputs
    val cachedDerivative: Array<Array<DTensor>?> = Array(gatheredOutputs.size) { null }
    fun extractOneDerivative(inputValue: DTensor, outputValue: DTensor): DTensor {
        val inputNum = gatheredInputs.indexOf(inputValue)
        val outputNum = gatheredOutputs.indexOf(outputValue)
        if (inputNum < 0 || outputNum < 0)
            return FloatTensor.zeros(inputValue.shape + outputValue.shape)
        var cached = cachedDerivative[outputNum]
        if (cached == null) {
            val derivative = when (outputValue) {
                is ForwardTensor -> {
                    require(outputValue.derivativeID == derivativeID)
                    outputValue.tangent
                }
                else -> throw IllegalArgumentException()
            }

            // derivative is of shape T<O,GI> where O is the shape of outputValue, GI is tangentShape
            val outputShape = outputValue.shape
            val dt = derivative.leftTranspose(outputShape, tangentShape) // shape T<GI,O>
            val split = dt.split(gatheredInputs.map { (it.shape + outputShape) as @SType("Shape") Shape }) // shape T<I,O> ...
            val splitT = split.mapIndexed { i: Int, t: DTensor -> t.leftTranspose(gatheredInputs[i].shape, outputShape) }
            cached = splitT.toTypedArray()
            cachedDerivative[inputNum] = cached
        }
        val result = cached[inputNum]
        assert(result.shape == outputValue.shape + inputValue.shape)
        return result
    }

    val tangentResult = extractDerivative(wrappedInput, wrappedOutput, ::extractOneDerivative)
    return Pair(unwrappedOutput, tangentResult)
}

fun <Input : Any, Output : Any, Derivative : Any> forwardDerivative(
    x: Input,
    f: (Input) -> Output,
    wrapInput: ((Input, Wrapper) -> Input)? = null,
    wrapOutput: ((Output, Wrapper) -> Output)? = null,
    extractDerivative: (input: Input, output: Output, extractOneDerivative: (input: DTensor, output: DTensor) -> DTensor) -> Derivative,
): Derivative {
    return primalAndForwardDerivative(x, f, wrapInput, wrapOutput, extractDerivative).second
}