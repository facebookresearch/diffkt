/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

import org.diffkt.reverse.ReverseDerivativeID
import org.diffkt.reverse.ReverseTensor

/**
 * Evaluate the function, and return a pair containing its result (primal) and the derivative.
 * This version supports user-defined types for input and output of the function.
 */
fun <Input : Any, Output : Any, Derivative : Any> primalAndReverseDerivative(
    x: Input,
    f: (Input) -> Output,
    wrapInput: ((Input, Wrapper) -> Input)? = null,
    wrapOutput: ((Output, Wrapper) -> Output)? = null,
    extractDerivative: (Input, Output, (input: DTensor, output: DTensor) -> DTensor) -> Derivative,
): Pair<Output, Derivative> {
    // Wrap the input
    val derivativeID = ReverseDerivativeID()

    // Make the inputs active variables
    val gatheredInputs: SequentialIntegerAssigner<ReverseTensor> = SequentialIntegerAssigner()
    val inputWrapper = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            return when (value) {
                is DScalar -> {
                    val rs = ActiveVariableReverseScalar(value, derivativeID)
                    gatheredInputs.add(rs)
                    rs
                }
                else -> {
                    val rt = ActiveVariableReverseTensor(value, derivativeID)
                    gatheredInputs.add(rt)
                    rt
                }
            }
        }
    }
    val wrappedInput = if (wrapInput != null) wrapInput(x, inputWrapper) else inputWrapper.wrap(x)

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
            return unwrapOutput(value)
        }
    }
    val unwrappedOutput = if (wrapOutput != null)
        wrapOutput(wrappedOutput, outputUnwrapper)
    else
        outputUnwrapper.wrap(wrappedOutput)

    // Backpropagate the derivatives
    val scalarGradient = gatheredOutputs.size == 0 || gatheredOutputs.size == 1 && gatheredOutputs[0] is DScalar
    if (gatheredOutputs.size == 0) {
        derivativeID.upstreamShape = Shape()
    } else {
        val meldedOutputs = meld(gatheredOutputs.values)
        if (meldedOutputs.derivativeID == derivativeID) {
            if (scalarGradient) {
                val gradientShape = Shape()
                derivativeID.upstreamShape = gradientShape
                val out = gatheredOutputs[0] as ReverseTensor
                val initialUpstream = meldedOutputs.primal.operations.identityGradientOfSameKind(meldedOutputs.primal, gradientShape)
                out.pushback(initialUpstream)
            } else {
                val gradientShape = meldedOutputs.shape
                derivativeID.upstreamShape = gradientShape
                meldedOutputs as ReverseTensor
                val initialUpstream = meldedOutputs.primal.operations.identityGradientOfSameKind(meldedOutputs.primal, gradientShape)
                meldedOutputs.pushback(initialUpstream)
            }
        } else {
            // Nothing to do... e.g. a constant output
            val gradientShape = meldedOutputs.shape
            derivativeID.upstreamShape = gradientShape
            assert(meldedOutputs.derivativeID.sequence < derivativeID.sequence)
        }
    }

    val gradientShape = derivativeID.upstreamShape
    // A special case for constant functions: we don't need to run the reverse pass.  All derivatives are zero.
    val map = if (gatheredOutputs.size != 0) derivativeID.reversePass() else HashMap<ReverseTensor, DTensor>()

    // Extract the derivative from the active input
    val cachedDerivative: Array<Array<DTensor>?> = Array(gatheredInputs.size) { null }
    fun extractTensorDerivative(inputValue: DTensor, outputValue: DTensor): DTensor {
        val inputNum = gatheredInputs.indexOf(inputValue)
        val outputNum = gatheredOutputs.indexOf(outputValue)
        if (inputNum < 0 || outputNum < 0)
            return FloatTensor.zeros(inputValue.shape + outputValue.shape)
        var cached = cachedDerivative[inputNum]
        if (cached == null) {
            val upstream = map.get(inputValue as ReverseTensor)!!

            // upstream is of shape T<I,GO> where I is the shape of inputValue, GO is the shape of meldedOutputs.
            require(upstream.shape == inputValue.shape + gradientShape) // TODO: for some reason assertions are disabled during development, so using require.
            val ct = upstream.leftTranspose(inputValue.shape, gradientShape) // shape T<GO,I>
            val split = if (scalarGradient)
                listOf(ct)
            else
                ct.split(gatheredOutputs.map { it.shape + inputValue.shape }) // shape T<O,I> ...
            val splitT = split.mapIndexed { i: Int, t: DTensor -> t.leftTranspose(gatheredOutputs[i].shape, inputValue.shape) }
            cached = splitT.toTypedArray()
            cachedDerivative[inputNum] = cached
        }
        val result = cached[outputNum]
        assert(result.shape == inputValue.shape + outputValue.shape)
        return result
    }

    val tangentResult = extractDerivative(wrappedInput, wrappedOutput, ::extractTensorDerivative)
    return Pair(unwrappedOutput, tangentResult)
}

fun <Input : Any, Output : Any, Derivative : Any> reverseDerivative(
    x: Input,
    f: (Input) -> Output,
    wrapInput: ((Input, Wrapper) -> Input)? = null,
    wrapOutput: ((Output, Wrapper) -> Output)? = null,
    extractDerivative: (Input, Output, (input: DTensor, output: DTensor) -> DTensor) -> Derivative,
): Derivative {
    return primalAndReverseDerivative(x, f, wrapInput, wrapOutput, extractDerivative).second
}