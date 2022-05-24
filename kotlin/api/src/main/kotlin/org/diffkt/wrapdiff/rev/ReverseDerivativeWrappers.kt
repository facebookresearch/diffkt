/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

// when Output == OutputTangent, we can use wrapOutput to infer setOutputTangent
internal fun <Output : Any> defaultSetOutputTangent(
    wrapOutput: ((Output, Wrapper) -> Output)? = null
): (Output, Output, setTensorTangent: (tensor: DTensor, tangent: DTensor) -> Unit) -> Unit {
    fun setOutputTangent(output: Output, tangent: Output, setTensorTangent: (tensor: DTensor, tangent: DTensor) -> Unit) {
        val gatheredTangents: MutableList<DTensor> = mutableListOf()

        val tangentGatherer = object : Wrapper() {
            override fun wrapDTensor(value: DTensor): DTensor {
                gatheredTangents.add(value)
                return value
            }
        }

        val tangentSetter = object : Wrapper() {
            var nextId = 0
            override fun wrapDTensor(value: DTensor): DTensor {
                val tangentTensor = gatheredTangents.getOrNull(nextId) ?:
                throw Exception("inconsistent number of leaves in tangent object, expected at least ${nextId + 1}")
                setTensorTangent(value, tangentTensor)
                nextId++
                return value
            }
        }

        if (wrapOutput != null) {
            wrapOutput(tangent, tangentGatherer)
            wrapOutput(output, tangentSetter)
        } else {
            tangentGatherer.wrap(tangent)
            tangentSetter.wrap(output)
        }

        val numTangentLeaves = gatheredTangents.size
        val numOutputLeaves = tangentSetter.nextId
        require(numOutputLeaves == numTangentLeaves) {
            "number of leaves in the output object (${numOutputLeaves}) is inconsistent with number of leaves in the tangent object (${numTangentLeaves})"
        }
    }

    return ::setOutputTangent
}

// when Input == InputTangent, we can use wrapInput to infer makeReverseInput and extractInputTangent
internal fun <Input : Any> defaultMakeReverseInput(
    wrapInput: ((Input, Wrapper) -> Input)? = null
): (Input, makeReverseTensor: (primal: DTensor) -> DTensor) -> Input {
    fun makeReverseInput(
        primal: Input, makeReverseTensor: (DTensor) -> DTensor,
    ): Input {
        val revWrapper = object : Wrapper() {
            override fun wrapDTensor(value: DTensor): DTensor {
                return makeReverseTensor(value)
            }
        }

        return if (wrapInput != null) {
            wrapInput(primal, revWrapper)
        } else {
            revWrapper.wrap(primal)
        }
    }
    return ::makeReverseInput
}

internal fun <Input : Any> defaultExtractInputTangent(
    wrapInput: ((Input, Wrapper) -> Input)? = null
): (Input, extractTensorTangent: (DTensor) -> DTensor) -> Input {
    fun extractInputTangent(
        input: Input, extractTensorTangent: (DTensor) -> DTensor
    ): Input {
        val tangentExtractor = object : Wrapper() {
            override fun wrapDTensor(value: DTensor): DTensor {
                return extractTensorTangent(value)
            }
        }

        return if (wrapInput != null) {
            wrapInput(input, tangentExtractor)
        } else {
            tangentExtractor.wrap(input)
        }
    }
    return ::extractInputTangent
}