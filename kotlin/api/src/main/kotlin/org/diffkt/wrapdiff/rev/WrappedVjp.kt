/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt

fun <Input : Any, Output : Any> primalAndVjp(
    x: Input,
    vf: (Output) -> Output,
    f: (x: Input) -> Output
): Pair<Output, Input> {
    val (primal, pullback) = primalAndPullback(x, f)
    val v = vf(primal)
    val vjp = pullback(v)
    return Pair(primal, vjp)
}

fun <Input : Any, Output : Any> vjp(
    x: Input,
    vf: (Output) -> Output,
    f: (x: Input) -> Output
): Input {
    return primalAndVjp(x, vf, f).second
}

fun <Input : Any, Output : Any> primalAndVjp(
    x: Input,
    v: Output,
    f: (x: Input) -> Output
): Pair<Output, Input> {
    val (primal, pullback) = primalAndPullback(x, f)
    val vjp = pullback(v)
    return Pair(primal, vjp)
}

fun <Input : Any, Output : Any> primalAndVjp(
    x: Input,
    v: Output,
    f: (x: Input) -> Output,
    makeReverseInput: (Input, makeReverseTensor: (DTensor) -> DTensor) -> Input,
    extractInputTangent: (Input, extractTensorTangent: (DTensor) -> DTensor) -> Input
): Pair<Output, Input> {
    val (primal, pullback) = primalAndPullback(
        x, f,
        makeReverseInput = makeReverseInput,
        extractInputTangent = extractInputTangent
    )
    val vjp = pullback(v)
    return Pair(primal, vjp)
}

fun <Input : Any, Output : Any, InputTangent : Any, OutputTangent : Any> primalAndVjp(
    x: Input,
    v: OutputTangent,
    f: (x: Input) -> Output,
    makeReverseInput: ((Input, makeReverseTensor: (DTensor) -> DTensor) -> Input)? = null,
    extractInputTangent: (Input, extractTensorTangent: (DTensor) -> DTensor) -> InputTangent,
    setOutputTangent: (Output, OutputTangent, setTensorTangent: (tensor: DTensor, tangent: DTensor) -> Unit) -> Unit,
    wrapOutput: ((Output, Wrapper) -> Output)? = null
): Pair<Output, InputTangent> {
    val (primal, pullback) = primalAndPullback(
        x, f,
        wrapOutput = wrapOutput,
        makeReverseInput = makeReverseInput ?: defaultMakeReverseInput<Input>(),
        setOutputTangent = setOutputTangent,
        extractInputTangent = extractInputTangent,
    )
    val vjp = pullback(v)
    return Pair(primal, vjp)
}

fun <Input : Any, Output : Any> vjp(
    x: Input,
    v: Output,
    f: (Input) -> Output
): Input {
    return primalAndVjp(x, v, f).second
}

// the gradient is a particular case of vjp where the output is a scalar and the input vector of the vjp is one
fun <Input : Any, InputTangent: Any> primalAndGradient(
    x: Input,
    f: (Input) -> DScalar,
    makeReverseInput: ((Input, makeReverseTensor: (DTensor) -> DTensor) -> Input)? = null,
    extractInputTangent: (Input, extractTensorTangent: (DTensor) -> DTensor) -> InputTangent
): Pair<DScalar, InputTangent> {
    return primalAndVjp(
        x, FloatScalar(1f),
        f,
        makeReverseInput = makeReverseInput ?: defaultMakeReverseInput<Input>(),
        extractInputTangent = extractInputTangent,
        setOutputTangent = defaultSetOutputTangent()
    )
}

// The `Input` type cannot be a strict subtype of DTensor; it must be DTensor or a non-DTensor Differentiable type.
fun <Input : Any> primalAndGradient(
    x: Input,
    f: (Input) -> DScalar,
): Pair<DScalar, Input> {
    return primalAndGradient(
        x, f,
        makeReverseInput = defaultMakeReverseInput<Input>(),
        extractInputTangent = defaultExtractInputTangent<Input>()
    )
}
