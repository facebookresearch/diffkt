/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import org.diffkt.*
import org.diffkt.random.RandomKey

/**
 * Print an arbitrary data structure by replacing any tracing tensors in it by a [PrintedTensor].
 * Also returns a string giving assignments that are needed to construct the value.
 * These assignments are needed to avoid duplicate evaluations if any tracing tensors
 * contain reused nodes.
 */
internal fun <TValue : Any> tracingPrintedForm(x: TValue, numInputs: Int): Pair<String, TValue> {
    // For the purpose of printing, we ignore trace IDs
    return tracingPrintedForm(dedag(x, numInputs, TraceId(), rewriteVariableReferences = false))
}

internal fun <TValue : Any> tracingPrintedForm(dedagged: DedaggedTracingTensor<TValue>): Pair<String, TValue> {
    val result = StringBuilder()
    for ((tempIndex, tempValue) in dedagged.assignments) {
        result.appendLine("val t$tempIndex = ${tempValue.rawPrintedForm()}")
    }
    val printingWrapper = object : Wrapper() {
        override fun wrapDTensor(value: DTensor): DTensor {
            return PrintedTensor(value.toCodeString(), value.shape)
        }

        override fun wrapRandomKey(value: RandomKey): RandomKey {
            return PrintedRandomKey(value.toString())
        }
    }
    return Pair(result.toString(), printingWrapper.wrap(dedagged.value))
}

/**
 * Print a tracing tensor, with reused nodes represented using an assignment to a temporary variable.
 */
fun Traceable.printedForm(numInputs: Int? = null): String {
    val allNodes = useCounts(listOf(this)).keys
    var inputCount = numInputs ?: 0
    for (node in allNodes) {
        if (node is TracingTensor.Variable) {
            // For the purpose of printing, we ignore trace IDs
            if (node.varIndex >= inputCount) inputCount = node.varIndex + 1
        }
    }
    // TODO: Account for randoms
    val (s, v) = tracingPrintedForm(this, inputCount)
    val final = v.rawPrintedForm()
    return s + final
}

/**
 * Print a tracing tensor, assuming there are no reused nodes.
 */
fun Traceable.rawPrintedForm() = when (this) {
    is PrintedTensor -> this.printed
    is PrintedRandomKey -> this.printed
    else -> TracingPrinter.print(this)
}

/**
 * This class is used when "printing" complex data structures by [tracingPrintedForm].  Tracing tensors
 * are replaced by an instance of this class to hold the printed form.
 */
class PrintedTensor(val printed: String, override val shape: Shape) : TracingTensor {
    override val derivativeID = NoDerivativeID
    override val primal = this
    override val operations: Operations get() = throw IllegalStateException("Cannot operate on printed tensors")
    override fun <R> accept(v: TracingVisitor<R>): R = throw IllegalStateException("Cannot operate on printed tensors")
    override fun toString() = printed
}

internal class PrintedRandomKey(val printed: String) : TracingRandomKey {
    override val traceId: TraceId = TraceId()
    override fun <R> accept(v: TracingVisitor<R>): R = throw IllegalStateException("Cannot operate on printed keys")
    override fun split(n: Int): List<RandomKey> = throw IllegalStateException("Cannot operate on printed keys")

    override fun floats(shape: Shape) = throw IllegalStateException("Cannot operate on printed keys")
    override fun permitReuse(): RandomKey = throw IllegalStateException("Cannot operate on printed keys")
    override fun gaussian(shape: Shape) = throw IllegalStateException("Cannot operate on printed keys")
    override fun gamma(alpha: FloatTensor) = throw IllegalStateException("Cannot operate on printed keys")
    override fun toString() = printed
}

/**
 * The result type of the tracing printer is a string containing the value, and a boolean which
 * is true if the resulting expression is not a primary and may need parens in an enclosing expression.
 */
typealias Result = Pair<String, Boolean>

/**
 * A visitor for producing the printed (string) form of tracing tensors that do not reuse values.
 */
private object TracingPrinter: TracingVisitor<Result> {
    fun print(x: Traceable): String {
        return visit(x).first
    }

    fun possiblyParens(x: TracingTensor): String {
        val p = visit(x)
        return if (p.second) "(${p.first})" else p.first
    }

    fun noParens(x: TracingTensor): String {
        val p = visit(x)
        return p.first
    }

    override fun visitConstant(x: TracingTensor.Constant): Result {
        return Pair(if (x.values is FloatScalar) "${x.values.value}f" else x.values.toCodeString(), false)
    }

    override fun visitVariable(x: TracingTensor.Variable): Result {
        return Pair(x.name ?: "t" + x.varIndex, false)
    }

    override fun visitPlus(x: TracingTensor.Plus): Result {
        return Pair(possiblyParens(x.left) + " + " + possiblyParens(x.right), true)
    }

    override fun visitMinus(x: TracingTensor.Minus): Result {
        return Pair(possiblyParens(x.left) + " - " + possiblyParens(x.right), true)
    }

    override fun visitTimes(x: TracingTensor.Times): Result {
        return Pair(possiblyParens(x.left) + " * " + possiblyParens(x.right), true)
    }

    override fun visitTimesScalar(x: TracingTensor.TimesScalar): Result {
        val left = x.left
        val right = x.right
        return if (right is TracingScalar.Constant && right.values is FloatScalar && left !is TracingScalar.Constant)
            Pair("${possiblyParens(right)} * ${possiblyParens(left)}", true)
        else
            Pair("${possiblyParens(left)} * ${possiblyParens(right)}", true)
    }

    override fun visitDiv(x: TracingTensor.Div): Result {
        return Pair(possiblyParens(x.left) + " / " + possiblyParens(x.right), true)
    }

    override fun visitZero(x: TracingTensor.Zero): Result {
        // TODO: can this be isolated from FloatTensor as one particular tensor implementation?
        return Pair("FloatTensor.zeros(${x.shape})", false)
    }

    override fun visitIdentityGradient(x: TracingTensor.IdentityGradient): Result {
        // TODO: can this be isolated from FloatTensor as one particular tensor implementation?
        return Pair("StridedFloatTensor.identityGradient(${x.halfShape})", false)
    }

    override fun visitUnaryMinus(x: TracingTensor.UnaryMinus): Result {
        return Pair("-" + possiblyParens(x.x), true)
    }

    override fun visitMatmul(x: TracingTensor.Matmul): Result {
        return Pair("${possiblyParens(x.x)}.matmul(${noParens(x.y)}, ${x.a}, ${x.b}, ${x.c}, ${x.d})", false)
    }

    override fun visitOuterProduct(x: TracingTensor.OuterProduct): Result {
        return Pair("${x.x}.outerProduct(${noParens(x.y)})", false)
    }

    override fun visitSin(x: TracingTensor.Sin): Result {
        return Pair("sin(${noParens(x.x)})", false)
    }

    override fun visitCos(x: TracingTensor.Cos): Result {
        return Pair("cos(${noParens(x.x)})", false)
    }

    override fun visitTan(x: TracingTensor.Tan): Result {
        return Pair("tan(${noParens(x.x)})", false)
    }

    override fun visitAtan(x: TracingTensor.Atan): Result {
        return Pair("atan(${noParens(x.x)})", false)
    }

    override fun visitExp(x: TracingTensor.Exp): Result {
        return Pair("exp(${noParens(x.x)})", false)
    }

    override fun visitLn(x: TracingTensor.Ln): Result {
        return Pair("ln(${noParens(x.x)})", false)
    }

    override fun visitLgamma(x: TracingTensor.Lgamma): Result {
        return Pair("lgamma(${noParens(x.x)})", false)
    }

    override fun visitDigamma(x: TracingTensor.Digamma): Result {
        return Pair("digamma(${noParens(x.x)})", false)
    }

    override fun visitPolygamma(x: TracingTensor.Polygamma): Result {
        return Pair("polygamma(${x.n}, ${noParens(x.x)})", false)
    }

    override fun visitSqrt(x: TracingTensor.Sqrt): Result {
        return Pair("sqrt(${noParens(x.x)})", false)
    }

    override fun visitTanh(x: TracingTensor.Tanh): Result {
        return Pair("tanh(${noParens(x.x)})", false)
    }

    override fun visitMeld(x: TracingTensor.Meld): Result {
        val s = x.values.map { noParens(it) }.joinToString { it }
        return Pair("meld($s)", false)
    }

    override fun visitSplit(x: TracingTensor.Split): Result {
        val s = x.shapes.joinToString { it.toString() }
        return Pair("${possiblyParens(x.x)}.split($s)", false)
    }

    override fun visitSplitPart(x: TracingTensor.SplitPart): Result {
        return Pair("${possiblyParens(x.from)}[${x.index}]", false)
    }

    override fun visitConcat(x: TracingTensor.Concat): Result {
        val s = x.slices.map { noParens(it) }.joinToString { it }
        return Pair("concat(listOf($s), axis = ${x.axis})", false)
    }

    override fun visitBroadcastTo(x: TracingTensor.BroadcastTo): Result {
        // we do not print broadcast operations, as those are implicit in source.
        // return Pair("${possiblyParens(x.x)}.broadcastTo(${x.shape})", false)
        return visit(x.x)
    }

    override fun visitConvImpl(x: TracingTensor.ConvImpl): Result {
        TODO("Not yet implemented")
    }

    override fun visitExpand(x: TracingTensor.Expand): Result {
        return Pair("${possiblyParens(x.x)}.expand(${x.shape})", false)
    }

    override fun visitFlip(x: TracingTensor.Flip): Result {
        TODO("Not yet implemented")
    }

    override fun visitLogSoftmax(x: TracingTensor.LogSoftmax): Result {
        return Pair("${possiblyParens(x.x)}.logSoftmax(${x.axis})", false)
    }

    override fun visitLogSoftmaxGrad(x: TracingTensor.LogSoftmaxGrad): Result {
        // pretend there is a top-level operation called logSoftmaxGrad
        return Pair("${possiblyParens(x.x)}.logSoftmaxGrad(${x.axis}, ${noParens(x.logSoftmax)}, ${noParens(x.upstream)})", false)
    }

    override fun visitPow(x: TracingTensor.Pow): Result {
        return Pair("${possiblyParens(x.base)}.pow(${x.exponent}f)", false)
    }

    override fun visitView1(x: TracingTensor.View1): Result {
        TODO("Not yet implemented")
    }

    override fun visitView2(x: TracingTensor.View2): Result {
        // tensor.view(index, axis)
        return if (x.axis == 0)
            Pair("${possiblyParens(x.x)}[${x.index}]", false)
        else
            Pair("${possiblyParens(x.x)}.view(${x.index}, axis = ${x.axis})", false)
    }

    override fun visitView3(x: TracingTensor.View3): Result {
        // DTensor.view(index: IntRange, axis: Int)
        return Pair("${possiblyParens(x.x)}.view(${x.index}, axis = ${x.axis})", false)
    }

    override fun visitReshape(x: TracingTensor.Reshape): Result {
        return Pair("${possiblyParens(x.x)}.reshape(${x.shape})", false)
    }

    override fun visitReshapeToScalar(x: TracingScalar.ReshapeToScalar): Result {
        return Pair("${possiblyParens(x.x)}.reshape(Shape())", false)
    }

    override fun visitSqueeze(x: TracingTensor.Squeeze): Result {
        return Pair("${possiblyParens(x.x)}.squeeze(axis = ${x.axis})", false)
    }

    override fun visitUnsqueeze(x: TracingTensor.Unsqueeze): Result {
        return Pair("${possiblyParens(x.x)}.unsqueeze(axis = ${x.axis})", false)
    }

    override fun visitTranspose(x: TracingTensor.Transpose): Result {
        TODO("Not yet implemented")
    }

    override fun visitRelu(x: TracingTensor.Relu): Result {
        return Pair("relu(${noParens(x.x)})", false)
    }

    override fun visitReluGrad(x: TracingTensor.ReluGrad): Result {
        return Pair("reluGrad(${noParens(x.x)}, ${noParens(x.upstream)})", false)
    }

    override fun visitSigmoid(x: TracingTensor.Sigmoid): Result {
        return Pair("sigmoid(${noParens(x.x)})", false)
    }

    override fun visitSum(x: TracingTensor.Sum): Result {
        return if (x.axes.size == x.x.rank && !x.keepDims)
            Pair("${possiblyParens(x.x)}.sum()", false)
        else
            Pair("${possiblyParens(x.x)}.sum(intArrayOf(${x.axes.joinToString(", ")}), keepDims = ${x.keepDims})", false)
    }

    override fun visitAvgPool(x: TracingTensor.AvgPool): Result {
        TODO("Not yet implemented")
    }

    override fun visitAvgPoolGrad(x: TracingTensor.AvgPoolGrad): Result {
        TODO("Not yet implemented")
    }

    override fun visitMaxPoolWithIndices(x: TracingTensor.MaxPoolWithIndices): Result {
        TODO("Not yet implemented")
    }

    override fun visitGather(x: TracingTensor.Gather): Result {
        return Pair("gather(${noParens(x.x)}, ${x.axis}, ${x.indices})", false)
    }

    override fun visitGatherAtIndices(x: TracingTensor.GatherAtIndices): Result {
        TODO("Not yet implemented")
    }

    override fun visitScatter(x: TracingTensor.Scatter): Result {
        return Pair("scatter(${noParens(x.x)}, ${x.axis}, ${x.indexes}, ${x.shape})", false)
    }

    override fun visitScatterAtIndices(x: TracingTensor.ScatterAtIndices): Result {
        TODO("Not yet implemented")
    }

    override fun visitCompare(x: TracingTensor.Compare): Result {
        return Pair("${possiblyParens(x.left)} ${x.comparison.toString().lowercase()} ${possiblyParens(x.right)}", true)
    }

    override fun visitIfThenElse(x: TracingTensor.IfThenElse): Result {
        return Pair("ifThenElse(${noParens(x.cond)}, ${noParens(x.whenTrue)}, ${noParens(x.whenFalse)})", false)
    }

    override fun visitRandomFloats(x: TracingTensor.RandomFloats): Result {
        return Pair("${x.key}.floats(${x.shape})", false)
    }

    override fun visitRandomSplit(x: TracingRandomKey.Split): Result = Pair("split", false)
    override fun visitRandomSplitPart(x: TracingRandomKey.SplitPart): Result = Pair("splitPart", false)
    override fun visitRandomVariable(x: TracingRandomKey.Variable): Result = Pair(x.name ?: "r${x.index}", false)
}
