/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import org.diffkt.*
import org.diffkt.random.RandomKey
import kotlin.IllegalArgumentException

/**
 * Transform a (differentiable) function into a function of the same signature but
 * which unrolls all loops and control constructs, and performs a set of local
 * optimizations on the result.  Because control-flow is removed,
 * any data-dependent control-flow in the function will no longer depend on the input
 * data in the second and subsequent invocations of the returned function.
 */
fun <TInput: Any, TOutput: Any> jit(f: (TInput) -> TOutput) : JittedFunction<TInput, TOutput> {
    return jit(f, evaluatorToUse = JitEvaluatorToUse.BestAvailable)
}

/**
 * The caller specifies which evaluator to use.
 * This is temporary; we want to select automatically.
 */
enum class JitEvaluatorToUse {
    None,
    Normal,
    Scalar,
    Jvm,
    BestAvailable
}

interface JittedFunction<TInput, TOutput> : (TInput) -> TOutput {
    val evaluatorToUse: JitEvaluatorToUse
    val cacheSize: Int
}

/**
 * Transform a (differentiable) function into a function of the same signature but
 * which unrolls all loops and control constructs, and performs a set of local
 * optimizations on the result.  Because control-flow is removed,
 * any data-dependent control-flow in the function will no longer depend on the input
 * data in the second and subsequent invocations of the returned function.
 */
fun <TInput: Any, TOutput: Any> jit(
    f: (TInput) -> TOutput,
    wrapInput: ((TInput, Wrapper) -> TInput)? = null,
    cacheSizeLimit: Int = 5,
    evaluatorToUse: JitEvaluatorToUse = JitEvaluatorToUse.BestAvailable,
    loggingName: String = "jit",
    shouldLog: Boolean = false,
): JittedFunction<TInput, TOutput> {
    data class JitCacheEntry<T : Any>(
        val output: DedaggedTracingTensor<T>,
        val optionalJvmFunction: ((FloatArray) -> T)?,
    )

    val evaluationLog = if (shouldLog) TimingStats("${loggingName}_timing") else null

    if (f is JittedFunction && f.evaluatorToUse == evaluatorToUse) return f
    if (evaluatorToUse == JitEvaluatorToUse.None) {
        return object : JittedFunction<TInput, TOutput> {
            override val evaluatorToUse = JitEvaluatorToUse.None
            override val cacheSize = 0
            override fun invoke(p1: TInput): TOutput {
                evaluationLog?.start()
                val result = f(p1)
                evaluationLog?.end()
                return result
            }
        }
    }

    return object : JittedFunction<TInput, TOutput> {
        val traceId = TraceId()
        override val evaluatorToUse get() = evaluatorToUse
        override val cacheSize get() = cache.size

        /**
         * This LRU cache stores the translation of the
         * input for each shape of input.  The result represents sharing by using
         * intermediate variables in the [DedaggedTracingTensor].
         */
        val cache = object : LinkedHashMap<TInput, JitCacheEntry<TOutput>>(
            cacheSizeLimit,
            /* loadFactor = */ 0.75f,
            /* accessOrder = */ true) {

            private val lock = Any() // private lock for synchronized block
            val cacheLog = if (shouldLog) FrequencyStats("${loggingName}_jit") else null
            override fun removeEldestEntry(eldest: MutableMap.MutableEntry<TInput, JitCacheEntry<TOutput>>?) =
                size > cacheSizeLimit

            override fun get(key: TInput): JitCacheEntry<TOutput>? {
                val result = synchronized(lock) { super.get(key) }
                cacheLog?.note(result != null)
                return result
            }

            override fun put(key: TInput, value: JitCacheEntry<TOutput>): JitCacheEntry<TOutput>? =
                synchronized(lock) { super.put(key, value) }

            override val size: Int get() = synchronized(lock) { super.size }
        }

        override fun invoke(input: TInput): TOutput {
            // When running the code, first we make a key for the cache based on the shape of the
            // tensors in the input to see if the code has already been processed for that shape of input.
            // We do that by constructing a new TInput with each tensor replaced by a tracing tensor.
            // At the same time, we make a mapping (in the form of a list) from variables index to
            // the actual input values to be used during evaluation
            evaluationLog?.start()
            var canEvalScalar = evaluatorToUse >= JitEvaluatorToUse.Scalar
            val variables = ArrayList<DTensor>()
            var nextInputIndex: Int = 0
            val tracingWrapper = object : Wrapper() {
                override fun wrapDTensor(value: DTensor): DTensor {
                    val index = nextInputIndex++
                    variables.add(value)
                    return when (value) {
                        is FloatScalar -> {
                            TracingScalar.Variable(index, traceId = traceId)
                        }
                        is DScalar -> {
                            canEvalScalar = false
                            TracingScalar.Variable(index, traceId = traceId)
                        }
                        else -> {
                            canEvalScalar = false
                            TracingTensor.Variable(index, shape = value.shape, traceId = traceId)
                        }
                    }
                }

                override fun wrapRandomKey(value: RandomKey): RandomKey {
                    canEvalScalar = false // Currently incompatible with JVM evaluator
                    val index = nextInputIndex++
                    variables.add(TracingRandomKey.RandomTensorWrapper(value))
                    return TracingRandomKey.Variable(index, traceId = traceId)
                }
            }

            if (evaluatorToUse >= JitEvaluatorToUse.Scalar && !canEvalScalar)
                throw IllegalArgumentException("Requested evaluatorToUse=${evaluatorToUse} but some input was not a FloatScalar")

            // Wrap the inputs in tracing tensors
            val tracingInput = if (wrapInput != null) wrapInput(input, tracingWrapper) else tracingWrapper.wrap(input)

            // Look for (or put) the translated code in the cache.
            var tracingResult =  cache.get(tracingInput)

            // If we don't find tracingInput (key) in the cache, we compute the traced result (value).
            if (tracingResult == null) {
                // Produce the tracing tensor result of the function for those shapes of inputs
                val result = f(tracingInput)
                // Simplify it
                val simplifiedResult = simplify(result)
                // Introduce temporary intermediate variables where values are reused or to reduce eval stack size
                val trace = dedeepen(dedag(simplifiedResult, nextInputIndex, traceId), depthLimit = 500)
                canEvalScalar = canEvalScalar && onlyScalarFloatValues(trace, traceId)
                if (evaluatorToUse >= JitEvaluatorToUse.Scalar && !canEvalScalar && evaluatorToUse != JitEvaluatorToUse.BestAvailable)
                    throw IllegalArgumentException("Requested evaluatorToUse=${evaluatorToUse} but some intermediate result was not a scalar")
                val jvmFunc =
                    if (canEvalScalar && (evaluatorToUse == JitEvaluatorToUse.Jvm || evaluatorToUse == JitEvaluatorToUse.BestAvailable)) {
                        val f = jvmTracingTranslator(trace)
                        if (f == null && evaluatorToUse == JitEvaluatorToUse.Jvm)
                            throw IllegalArgumentException("Requested jvm evaluator but the function would be too large.")
                        f
                    } else null
                tracingResult = JitCacheEntry(trace, jvmFunc)
                // Place the now optimized tracing tensor in the cache in case we ever see these
                // shapes of inputs again in the future.
                cache.put(tracingInput, tracingResult)
            }

            val (trace, jvmEval) = tracingResult
            val result = when {
                !canEvalScalar || !trace.canScalarEval -> eval(trace, variables)
                jvmEval != null -> {
                    val floatVariables = variables.map { (it as FloatScalar).value }.toFloatArray()
                    jvmEval(floatVariables)
                }
                else -> scalarEval(trace, variables)
            }
            evaluationLog?.end()
            return result
        }

        fun onlyScalarFloatValues(trace: TracingTensor, traceId: TraceId): Boolean {
            for (t in topologicalSort(listOf(trace), skip = { false })) {
                if (t is TracingTensor && t.shape != Shape()) return false
                when (t) {
                    is TracingScalar.Variable -> if (t.traceId != traceId) return false
                    is TracingScalar.Constant -> if (t.values !is FloatScalar) return false
                    is TracingTensor.RandomFloats -> if (t.traceId != traceId) return false
                }
            }

            return true
        }

        fun <TOutput: Any> onlyScalarFloatValues(trace: DedaggedTracingTensor<TOutput>, traceId: TraceId): Boolean {
            var result = true
            for (a in trace.assignments) {
                if (a.second is TracingTensor) {
                    result = result && onlyScalarFloatValues(a.second as TracingTensor, traceId)
                }
            }
            val resultScanner = object : Wrapper() {
                override fun wrapDTensor(value: DTensor): DTensor {
                    result = result && (value is FloatScalar || value is TracingTensor && onlyScalarFloatValues(value, traceId))
                    return value
                }
            }
            resultScanner.wrap(trace.value)
            return result
        }
    }
}

internal fun <TValue: Any> scalarEval(tracingResult: DedaggedTracingTensor<TValue>, variables: List<DTensor>): TValue {
    val floatVariables = FloatArray(tracingResult.numInputs + tracingResult.numTemps)
    var nextInputIndex = 0
    variables.forEach() { value ->
        floatVariables[nextInputIndex++] = (value as FloatScalar).value
    }
    assert(nextInputIndex == tracingResult.numInputs)

    // Evaluate the assignments to the intermediate temporary variables in the optimized form
    var nextTempIndex = tracingResult.numInputs
    tracingResult.assignments.forEach {
        val (tempIndex, tempValue) = it
        assert(tempIndex == nextTempIndex)
        val value = (tempValue as TracingTensor).floatEval(floatVariables)
        floatVariables[tempIndex] = value
        nextTempIndex++
    }

    // Evaluate the final output.
    val outputEvaluator = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            return when (value) {
                is TracingTensor -> FloatScalar(value.floatEval(floatVariables))
                else -> value
            }
        }
    }
    return outputEvaluator.wrap(tracingResult.value)
}

/**
 * Jitting evaluator for testing purposes only.  Creates a jitted function and evaluates it.
 * It is only useful for testing because the generated function is discarded after being evaluated.
 * After all the work of generating it, in real programs you would save the jvm function for reuse.
 */
internal fun <TValue: Any> jitEval(tracingResult: DedaggedTracingTensor<TValue>, variables: List<DTensor>): TValue {
    val jf = jvmTracingTranslator(tracingResult)
    val input = variables.map { (it as FloatScalar).value }.toFloatArray()
    assert(jf != null)
    return jf!!.invoke(input)
}

internal fun <TValue: Any> eval(tracingResult: DedaggedTracingTensor<TValue>, variables: List<DTensor>): TValue {
    val traceId = tracingResult.traceId
    val tensorVariables = Array<DTensor?>(tracingResult.numInputs + tracingResult.numTemps) { null }
    var nextInputIndex = 0
    variables.forEach() { value ->
        tensorVariables[nextInputIndex++] = value
    }
    assert(nextInputIndex == tracingResult.numInputs)

    // Evaluate the assignments to the intermediate temporary variables in the optimized form
    val evaluator = TracingEvaluator(tensorVariables, traceId)
    var nextTempIndex = tracingResult.numInputs
    tracingResult.assignments.forEach {
        val (tempIndex, tempValue) = it
        assert(tempIndex == nextTempIndex)
        val value = evaluator.evaluate(tempValue)
        tensorVariables[tempIndex] = value
        nextTempIndex++
    }

    // Evaluate the final output.
    val outputEvaluator = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            return when (value) {
                is TracingTensor -> evaluator.evaluate(value)
                else -> value
            }
        }

        override fun wrapRandomKey(value: RandomKey): RandomKey {
            return when (value) {
                is TracingRandomKey -> (evaluator.evaluate(value) as TracingRandomKey.RandomTensorWrapper).key
                else -> value
            }
        }
    }
    return outputEvaluator.wrap(tracingResult.value)
}
