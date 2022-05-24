/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import java.util.*
import org.diffkt.*

/**
 * A topological sort, which processes an acyclic graph and returns
 * a topologically sorted list of its nodes, in which each node precedes any appearance of its
 * successors.  Returns null if the input graph is found to have a cycle.
 */
fun <TNode: Any?> topologicalSort(
    roots: List<TNode>,
    successors: (TNode) -> List<TNode>,
    skip: (TNode) -> Boolean = { false }): List<TNode>? {

    // First, count the predecessors of each node
    val predecessorCounts: HashMap<TNode, Int> = predecessorCounts(roots, successors, skip)

    // Initialize the ready set with those nodes that have no predecessors
    val ready = Stack<TNode>()
    for ((k, v) in predecessorCounts) {
        if (v == 0)
            ready.push(k)
    }

    // Process the ready set. Output a node, and decrement the predecessor count of its successors.
    val result = Stack<TNode>()
    while (!ready.isEmpty())
    {
        val node = ready.pop()
        result.add(node)
        for (succ in successors(node)) {
            if (skip(succ)) continue
            val count = predecessorCounts[succ]!!
            assert(count != 0)
            predecessorCounts[succ] = count - 1
            if (count == 1)
                ready.push(succ)
        }
    }

    // At this point all the nodes should have been output, otherwise there was a cycle
    val hadCycle: Boolean = predecessorCounts.size != result.size
    return if (hadCycle) null else result
}

private fun <TNode: Any?> predecessorCounts(
    roots: List<TNode>,
    successors: (TNode) -> List<TNode>,
    skip: (TNode) -> Boolean = { false }): HashMap<TNode, Int> {
    val predecessorCounts = HashMap<TNode, Int>()
    val counted = HashSet<TNode>()
    val toCount = Stack<TNode>()
    for (r in roots) {
        if (!skip(r))
            toCount.add(r)
    }

    while (!toCount.isEmpty()) {
        val n = toCount.pop();
        if (!counted.add(n))
            continue

        if (!predecessorCounts.containsKey(n))
            predecessorCounts.put(n, 0)

        for (succ in successors(n)) if (!skip(succ)) {
            toCount.push(succ)
            if (predecessorCounts.containsKey(succ))
                predecessorCounts[succ] = predecessorCounts[succ]!! + 1
            else
                predecessorCounts.put(succ, 1)
        }
    }

    return predecessorCounts
}

fun useCounts(roots: List<Traceable>): HashMap<Traceable, Int> {
    val result = predecessorCounts(roots, ::children, { false })
    // Also count appearences in roots as a predecessor.
    for (r in roots)
        result[r] = result[r]!! + 1
    return result
}

/**
 * A topological sort of a set of tracing tensors, in which each tensor appears
 * after any of its inputs.
 */
internal fun topologicalSort(
    roots: List<Traceable>,
    skip: (Traceable) -> Boolean): List<Traceable> {
    return topologicalSort(roots, ::children, skip)!!
}

internal fun children(x: Traceable) = x.accept(childrenVisitor)

private object childrenVisitor: TracingVisitor<List<Traceable>> {
    override fun visitConstant(x: TracingTensor.Constant) = listOf<TracingTensor>()
    override fun visitVariable(x: TracingTensor.Variable) = listOf<TracingTensor>()
    override fun visitPlus(x: TracingTensor.Plus) = listOf(x.left, x.right)
    override fun visitMinus(x: TracingTensor.Minus) = listOf(x.left, x.right)
    override fun visitTimes(x: TracingTensor.Times) = listOf(x.left, x.right)
    override fun visitTimesScalar(x: TracingTensor.TimesScalar) = listOf(x.left, x.right)
    override fun visitDiv(x: TracingTensor.Div) = listOf(x.left, x.right)
    override fun visitZero(x: TracingTensor.Zero) = listOf<TracingTensor>()
    override fun visitIdentityGradient(x: TracingTensor.IdentityGradient) = listOf<TracingTensor>()
    override fun visitUnaryMinus(x: TracingTensor.UnaryMinus) = listOf(x.x)
    override fun visitMatmul(x: TracingTensor.Matmul) = listOf(x.x, x.y)
    override fun visitOuterProduct(x: TracingTensor.OuterProduct) = listOf(x.x, x.y)
    override fun visitSin(x: TracingTensor.Sin) = listOf(x.x)
    override fun visitCos(x: TracingTensor.Cos) = listOf(x.x)
    override fun visitTan(x: TracingTensor.Tan) = listOf(x.x)
    override fun visitAtan(x: TracingTensor.Atan) = listOf(x.x)
    override fun visitExp(x: TracingTensor.Exp) = listOf(x.x)
    override fun visitLn(x: TracingTensor.Ln) = listOf(x.x)
    override fun visitLgamma(x: TracingTensor.Lgamma) = listOf(x.x)
    override fun visitDigamma(x: TracingTensor.Digamma) = listOf(x.x)
    override fun visitPolygamma(x: TracingTensor.Polygamma) = listOf(x.x)
    override fun visitSqrt(x: TracingTensor.Sqrt) = listOf(x.x)
    override fun visitTanh(x: TracingTensor.Tanh) = listOf(x.x)
    override fun visitMeld(x: TracingTensor.Meld) = x.values
    override fun visitSplit(x: TracingTensor.Split) = listOf(x.x)
    override fun visitSplitPart(x: TracingTensor.SplitPart) = listOf(x.from)
    override fun visitConcat(x: TracingTensor.Concat) = x.slices
    override fun visitBroadcastTo(x: TracingTensor.BroadcastTo) = listOf(x.x)
    override fun visitConvImpl(x: TracingTensor.ConvImpl) = listOf(x.filter, x.signal)
    override fun visitExpand(x: TracingTensor.Expand) = listOf(x.x)
    override fun visitFlip(x: TracingTensor.Flip) = listOf(x.x)
    override fun visitLogSoftmax(x: TracingTensor.LogSoftmax) = listOf(x.x)
    override fun visitLogSoftmaxGrad(x: TracingTensor.LogSoftmaxGrad) = listOf(x.x, x.logSoftmax, x.upstream)
    override fun visitPow(x: TracingTensor.Pow) = listOf(x.base)
    override fun visitView1(x: TracingTensor.View1) = listOf(x.x)
    override fun visitView2(x: TracingTensor.View2) = listOf(x.x)
    override fun visitView3(x: TracingTensor.View3) = listOf(x.x)
    override fun visitReshape(x: TracingTensor.Reshape) = listOf(x.x)
    override fun visitReshapeToScalar(x: TracingScalar.ReshapeToScalar) = listOf(x.x)
    override fun visitSqueeze(x: TracingTensor.Squeeze) = listOf(x.x)
    override fun visitUnsqueeze(x: TracingTensor.Unsqueeze) = listOf(x.x)
    override fun visitTranspose(x: TracingTensor.Transpose) = listOf(x.x)
    override fun visitRelu(x: TracingTensor.Relu) = listOf(x.x)
    override fun visitReluGrad(x: TracingTensor.ReluGrad) = listOf(x.x, x.upstream)
    override fun visitSigmoid(x: TracingTensor.Sigmoid) = listOf(x.x)
    override fun visitSum(x: TracingTensor.Sum) = listOf(x.x)
    override fun visitAvgPool(x: TracingTensor.AvgPool) = listOf(x.x)
    override fun visitAvgPoolGrad(x: TracingTensor.AvgPoolGrad) = listOf(x.x)
    override fun visitMaxPoolWithIndices(x: TracingTensor.MaxPoolWithIndices) = listOf(x.x)
    override fun visitGather(x: TracingTensor.Gather) = listOf(x.x)
    override fun visitGatherAtIndices(x: TracingTensor.GatherAtIndices) = listOf(x.x)
    override fun visitScatter(x: TracingTensor.Scatter) = listOf(x.x)
    override fun visitScatterAtIndices(x: TracingTensor.ScatterAtIndices) = listOf(x.x)
    override fun visitCompare(x: TracingTensor.Compare) = listOf(x.left, x.right)
    override fun visitIfThenElse(x: TracingTensor.IfThenElse) = listOf(x.cond, x.whenTrue, x.whenFalse)
    override fun visitRandomFloats(x: TracingTensor.RandomFloats) = listOf(x.key)
    override fun visitRandomVariable(x: TracingRandomKey.Variable): List<Traceable> = listOf()
    override fun visitRandomSplit(x: TracingRandomKey.Split): List<Traceable> = listOf(x.key)
    override fun visitRandomSplitPart(x: TracingRandomKey.SplitPart): List<Traceable> = listOf(x.split)
}
