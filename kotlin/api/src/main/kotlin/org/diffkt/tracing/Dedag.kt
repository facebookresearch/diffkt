/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import org.diffkt.*
import org.diffkt.random.RandomKey
import java.lang.IllegalStateException
import java.lang.Integer.max

/**
 * The result of removing reused nodes in a data structure.
 * Reused nodes are replaced by a [TracingTensor.Variable], and an assignment
 * to that variable is placed into the resulting [assignments].
 */
data class DedaggedTracingTensor<T: Any>(
    val numInputs: Int,
    val numTemps: Int,
    val numResults: Int,
    val assignments: List<Pair<Int, Traceable>>,
    val value: T,
    val traceId: TraceId,
    val canScalarEval: Boolean,
)

/**
 * Dedag (remove reused tracing tensor nodes from) an arbitrary data structure.
 */
fun <T: Any> dedag(value: T, numInputs: Int, traceId: TraceId, rewriteVariableReferences: Boolean = true): DedaggedTracingTensor<T> {
    val reused = reusedNodes(value)
    val assignments = ArrayList<Pair<Int, Traceable>>()

    // A rewriter that removes reused nodes and moves them into the assignments. Also determines whether the trace can be scalar evaluated.
    var numTemps = 0
    var canScalarEval = true
    val rewriter = object : DeepTracingRewriter() {
        // number temporary variables starting at numInputs
        private fun canScalarEvaluate(x: Traceable): Boolean {
            if (x is TracingTensor && !x.isScalar) return false
            if (x is TracingTensor.Variable) return x.traceId == traceId
            if (x is TracingTensor.RandomFloats) return x.shape.isScalar && x.traceId == traceId
            return true
        }
        private fun shouldRewriteAsVar(x: Traceable, rewrittenValue: Traceable): Boolean {
            return reused.contains(x) && when (rewrittenValue) {
                is TracingRandomKey.Variable -> rewriteVariableReferences && rewrittenValue.traceId != traceId
                is TracingTensor.Variable -> rewriteVariableReferences && rewrittenValue.traceId != traceId
                is TracingTensor.Constant -> false
                else -> true
            }
        }

        override fun rewriteOne(x: Traceable): Traceable {
            val rewrittenValue = super.rewriteOne(x)
            if (!canScalarEvaluate(x)) {
                canScalarEval = false
            }

            if (shouldRewriteAsVar(x, rewrittenValue)) {
                val tempIndex = numInputs + numTemps
                val variable = rewriteAsVariable(x, tempIndex, traceId)
                assignments.add(Pair(tempIndex, rewrittenValue))
                numTemps++
                return variable
            }
            return rewrittenValue
        }
    }
    // A value wrapper that invokes the rewriter to remove reused tracing tensor nodes
    var numResults = 0
    val valueWrapper = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            return when (value) {
                is TracingTensor -> {
                    numResults++
                    rewriter.rewrite(value) as TracingTensor
                }
                else -> value
            }
        }

        override fun wrapRandomKey(value: RandomKey): RandomKey {
            return when (value) {
                is TracingRandomKey -> {
                    numResults++
                    rewriter.rewrite(value) as TracingRandomKey
                }
                else -> value
            }
        }
    }

    val result = valueWrapper.wrap(value)
    return DedaggedTracingTensor(numInputs, numTemps, numResults, assignments, result, traceId, canScalarEval)
}

/**
 * Rewrite a dedagged result to remove deep recursion.
 */
internal fun <T: Any> dedeepen(input: DedaggedTracingTensor<T>, depthLimit: Int = 250): DedaggedTracingTensor<T> {
    val assignments = ArrayList<Pair<Int, Traceable>>()
    val numInputs = input.numInputs
    var numTemps = input.numTemps

    val depthCalculator = DepthCalculator()
    val transformer = object: DeepTracingRewriter() {
        override fun rewriteOne(x: Traceable): Traceable {
            val x2 = super.rewriteOne(x)
            val depth = depthCalculator.depth(x2)
            if (depth >= depthLimit) {
                val newTempNumber = numInputs + numTemps++
                val newTemp = rewriteAsVariable(x, newTempNumber, input.traceId, x2)
                assignments.add(Pair(newTempNumber, x2))
                return newTemp
            }

            return x2
        }
    }

    for ((temp, tempValue) in input.assignments) {
        val shallowTempValue = transformer.rewrite(tempValue)
        assignments.add(Pair(temp, shallowTempValue))
    }

    val valueWrapper = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            return when (value) {
                is TracingTensor -> transformer.rewrite(value) as TracingTensor
                else -> value
            }
        }

        override fun wrapRandomKey(value: RandomKey): RandomKey {
            return when (value) {
                is TracingRandomKey -> transformer.rewrite(value) as TracingRandomKey
                else -> value
            }
        }
    }

    val shallowResult = valueWrapper.wrap(input.value)
    return DedaggedTracingTensor(input.numInputs, numTemps, input.numResults, assignments, shallowResult, input.traceId, input.canScalarEval)
}

internal fun rewriteAsVariable(x: Traceable, idx: Int, traceId: TraceId, rewritten: Traceable = x): Traceable = when (x) {
    is TracingTensor -> if (x.isScalar) TracingScalar.Variable(idx, traceId = traceId)
        else TracingTensor.Variable(idx, shape = (rewritten as TracingTensor).shape, traceId = traceId)
    is TracingRandomKey -> TracingRandomKey.Variable(idx, traceId)
    else -> throw NotImplementedError("Dedag rewrites not yet supported for traceable ${x::class}")
}

private class DepthCalculator: TracingVisitor<Int> {
    private val depthMap = HashMap<Traceable, Int>()

    fun depth(x: Traceable): Int {
        val result = depthMap[x]
        if (result != null) return result
        val computed = x.accept(this)
        depthMap.put(x, computed)
        return computed
    }

    override fun visitConstant(x: TracingTensor.Constant) = 1
    override fun visitVariable(x: TracingTensor.Variable) = 1
    override fun visitPlus(x: TracingTensor.Plus) = 1 + max(depth(x.left), depth(x.right))
    override fun visitMinus(x: TracingTensor.Minus) = 1 + max(depth(x.left), depth(x.right))
    override fun visitTimes(x: TracingTensor.Times) = 1 + max(depth(x.left), depth(x.right))
    override fun visitTimesScalar(x: TracingTensor.TimesScalar) = 1 + max(depth(x.left), depth(x.right))
    override fun visitDiv(x: TracingTensor.Div) = 1 + max(depth(x.left), depth(x.right))
    override fun visitZero(x: TracingTensor.Zero) = 1
    override fun visitIdentityGradient(x: TracingTensor.IdentityGradient) = 1
    override fun visitUnaryMinus(x: TracingTensor.UnaryMinus) = 1 + depth(x.x)
    override fun visitMatmul(x: TracingTensor.Matmul) = 1 + max(depth(x.x), depth(x.y))
    override fun visitOuterProduct(x: TracingTensor.OuterProduct) = 1 + max(depth(x.x), depth(x.y))
    override fun visitSin(x: TracingTensor.Sin) = 1 + depth(x.x)
    override fun visitCos(x: TracingTensor.Cos) = 1 + depth(x.x)
    override fun visitTan(x: TracingTensor.Tan) = 1 + depth(x.x)
    override fun visitAtan(x: TracingTensor.Atan) = 1 + depth(x.x)
    override fun visitExp(x: TracingTensor.Exp) = 1 + depth(x.x)
    override fun visitLn(x: TracingTensor.Ln) = 1 + depth(x.x)
    override fun visitLgamma(x: TracingTensor.Lgamma) = 1 + depth(x.x)
    override fun visitDigamma(x: TracingTensor.Digamma) = 1 + depth(x.x)
    override fun visitPolygamma(x: TracingTensor.Polygamma) = 1 + depth(x.x)
    override fun visitSqrt(x: TracingTensor.Sqrt) = 1 + depth(x.x)
    override fun visitTanh(x: TracingTensor.Tanh) = 1 + depth(x.x)
    override fun visitMeld(x: TracingTensor.Meld) = 1 + x.values.map { depth(it) }.fold(0) { a, i -> max(a, i) }
    override fun visitSplit(x: TracingTensor.Split) = 1 + depth(x.x)
    override fun visitSplitPart(x: TracingTensor.SplitPart) = 1 + depth(x.from)
    override fun visitConcat(x: TracingTensor.Concat) = 1 + x.slices.map { depth(it) }.fold(0) { a, i -> max(a, i) }
    override fun visitBroadcastTo(x: TracingTensor.BroadcastTo) = 1 + depth(x.x)
    override fun visitConvImpl(x: TracingTensor.ConvImpl) = 1 + max(depth(x.filter), depth(x.signal))
    override fun visitExpand(x: TracingTensor.Expand) = 1 + depth(x.x)
    override fun visitFlip(x: TracingTensor.Flip) = 1 + depth(x.x)
    override fun visitLogSoftmax(x: TracingTensor.LogSoftmax) = 1 + depth(x.x)
    override fun visitLogSoftmaxGrad(x: TracingTensor.LogSoftmaxGrad) = 1 + max(depth(x.x), max(depth(x.logSoftmax), depth(x.upstream)))
    override fun visitPow(x: TracingTensor.Pow) = 1 + depth(x.base)
    override fun visitView1(x: TracingTensor.View1) = 1 + depth(x.x)
    override fun visitView2(x: TracingTensor.View2) = 1 + depth(x.x)
    override fun visitView3(x: TracingTensor.View3) = 1 + depth(x.x)
    override fun visitReshape(x: TracingTensor.Reshape) = 1 + depth(x.x)
    override fun visitReshapeToScalar(x: TracingScalar.ReshapeToScalar) = 1 + depth(x.x)
    override fun visitSqueeze(x: TracingTensor.Squeeze) = 1 + depth(x.x)
    override fun visitUnsqueeze(x: TracingTensor.Unsqueeze) = 1 + depth(x.x)
    override fun visitTranspose(x: TracingTensor.Transpose) = 1 + depth(x.x)
    override fun visitRelu(x: TracingTensor.Relu) = 1 + depth(x.x)
    override fun visitReluGrad(x: TracingTensor.ReluGrad) = 1 + max(depth(x.x), depth(x.upstream))
    override fun visitSigmoid(x: TracingTensor.Sigmoid) = 1 + depth(x.x)
    override fun visitSum(x: TracingTensor.Sum) = 1 + depth(x.x)
    override fun visitAvgPool(x: TracingTensor.AvgPool) = 1 + depth(x.x)
    override fun visitAvgPoolGrad(x: TracingTensor.AvgPoolGrad) = 1 + depth(x.x)
    override fun visitMaxPoolWithIndices(x: TracingTensor.MaxPoolWithIndices) = 1 + depth(x.x)
    override fun visitGather(x: TracingTensor.Gather) = 1 + depth(x.x)
    override fun visitGatherAtIndices(x: TracingTensor.GatherAtIndices) = 1 + depth(x.x)
    override fun visitScatter(x: TracingTensor.Scatter) = 1 + depth(x.x)
    override fun visitScatterAtIndices(x: TracingTensor.ScatterAtIndices) = 1 + depth(x.x)
    override fun visitCompare(x: TracingTensor.Compare) = 1 + max(depth(x.left), depth(x.right))
    override fun visitIfThenElse(x: TracingTensor.IfThenElse) = 1 + max(max(depth(x.cond), depth(x.whenTrue)), depth(x.whenFalse))
    override fun visitRandomFloats(x: TracingTensor.RandomFloats): Int = 1 + depth(x.key)
    override fun visitRandomSplit(x: TracingRandomKey.Split): Int = 1 + depth(x.key)
    override fun visitRandomSplitPart(x: TracingRandomKey.SplitPart): Int = 1 + depth(x.split)
    override fun visitRandomVariable(x: TracingRandomKey.Variable): Int = 1
}