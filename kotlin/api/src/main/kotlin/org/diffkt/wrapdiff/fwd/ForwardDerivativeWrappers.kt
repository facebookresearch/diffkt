/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

// when Input == InputTangent, we can use wrapInput to infer makeForwardInput
internal fun <Input : Any> defaultMakeForwardInput(
    wrapInput: ((Input, Wrapper) -> Input)? = null
): (primal: Input, tangent: Input, makeForwardTensor: (primal: DTensor, tangent: DTensor) -> DTensor) -> Input {
    fun makeForwardInput(
        primal: Input, tangent: Input,
        makeForwardTensor: (primal: DTensor, tangent: DTensor) -> DTensor
    ): Input {
        val gatheredTangents: MutableList<DTensor> = mutableListOf()

        val tangentGatherer = object : Wrapper() {
            override fun wrapDTensor(value: DTensor): DTensor {
                gatheredTangents.add(value)
                return value
            }
        }

        val fwdWrapper = object : Wrapper() {
            var nextId = 0
            override fun wrapDTensor(value: DTensor): DTensor {
                val tangentTensor = gatheredTangents.getOrNull(nextId) ?:
                throw Exception("inconsistent number of leaves in tangent object, expected at least ${nextId + 1}")
                val fwdTensor = makeForwardTensor(value, tangentTensor)
                nextId++
                return fwdTensor
            }
        }

        val fwdInput = if (wrapInput != null) {
            wrapInput(tangent, tangentGatherer)
            wrapInput(primal, fwdWrapper)
        } else {
            tangentGatherer.wrap(tangent)
            fwdWrapper.wrap(primal)
        }

        val numTangentLeaves = gatheredTangents.size
        val numPrimalLeaves = fwdWrapper.nextId
        require(numPrimalLeaves == numTangentLeaves) {
            "number of leaves in the primal object (${numPrimalLeaves}) is inconsistent with number of leaves in the tangent object (${numTangentLeaves})"
        }

        return fwdInput
    }

    return ::makeForwardInput
}

// when Output == OutputTangent, we can use wrapOutput to infer extractOutputTangent
internal fun <Output : Any> defaultExtractOutputTangent(
    wrapOutput: ((Output, Wrapper) -> Output)? = null
): (Output, extractTensorTangent: (outputTensor: DTensor) -> DTensor) -> Output {
    fun extractOutputTangent(
        output: Output, extractTensorTangent: (DTensor) -> DTensor
    ): Output {
        val tangentExtractor = object : Wrapper() {
            override fun wrapDTensor(value: DTensor): DTensor {
                return extractTensorTangent(value)
            }
        }

        return if (wrapOutput != null) {
            wrapOutput(output, tangentExtractor)
        } else {
            tangentExtractor.wrap(output)
        }
    }

    return ::extractOutputTangent
}