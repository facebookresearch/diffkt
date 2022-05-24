/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.diffkt.tracing

import org.diffkt.DScalar
import org.diffkt.DTensor

internal open class DeepTracingRewriter : ShallowTracingRewriter() {
    val rewrittenForm = HashMap<Traceable, Traceable>()

    fun rewrite(x: Traceable): Traceable {
        for (n in topologicalSort(listOf(x), skip = { rewrittenForm.containsKey(it) }).reversed()) {
            val rewritten = rewriteOne(n)
            rewrittenForm.put(n, rewritten)
        }

        return rewrittenForm(x)
    }

    open fun rewriteOne(x: Traceable): Traceable {
        val r = rewrittenForm.get(x)
        if (r != null)
            return r

        var rewritten = x.accept(this)
        return rewritten
    }

    private fun rewrittenForm(x: Traceable): Traceable {
        return rewrittenForm[x]!!
    }

    private fun rewrittenForm(x: TracingTensor): TracingTensor = rewrittenForm[x] as TracingTensor
    private fun rewrittenForm(x: TracingRandomKey): TracingRandomKey = rewrittenForm[x] as TracingRandomKey

    override fun visitPlus(x: TracingTensor.Plus): TracingTensor {
        val left = rewrittenForm(x.left)
        val right = rewrittenForm(x.right)
        return if (left !== x.left || right !== x.right)
            if (left is TracingScalar && right is TracingScalar) TracingScalar.Plus(left, right) else TracingTensor.Plus(left, right)
        else
            x
    }

    override fun visitMinus(x: TracingTensor.Minus): TracingTensor {
        val left = rewrittenForm(x.left)
        val right = rewrittenForm(x.right)
        return if (left !== x.left || right !== x.right)
            if (left is TracingScalar && right is TracingScalar) TracingScalar.Minus(left, right) else TracingTensor.Minus(left, right)
        else
            x
    }

    override fun visitTimes(x: TracingTensor.Times): TracingTensor {
        val left = rewrittenForm(x.left)
        val right = rewrittenForm(x.right)
        return if (left !== x.left || right !== x.right)
            TracingTensor.Times(left, right)
        else
            x
    }

    override fun visitTimesScalar(x: TracingTensor.TimesScalar): TracingTensor {
        val left = rewrittenForm(x.left) as TracingScalar
        val right = rewrittenForm(x.right)
        return if (left !== x.left || right !== x.right)
            if (right is TracingScalar) TracingScalar.TimesScalar(left, right) else TracingTensor.TimesScalar(left, right)
        else
            x
    }

    override fun visitDiv(x: TracingTensor.Div): TracingTensor {
        val left = rewrittenForm(x.left)
        val right = rewrittenForm(x.right)
        return if (left !== x.left || right !== x.right)
            if (left is TracingScalar && right is TracingScalar) TracingScalar.Div(left, right) else TracingTensor.Div(left, right)
        else
            x
    }

    override fun visitZero(x: TracingTensor.Zero): TracingTensor {
        return x
    }

    override fun visitIdentityGradient(x: TracingTensor.IdentityGradient): TracingTensor {
        return x
    }

    override fun visitUnaryMinus(x: TracingTensor.UnaryMinus): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.UnaryMinus(xx) else TracingTensor.UnaryMinus(xx)
        else
            x
    }

    override fun visitMatmul(x: TracingTensor.Matmul): TracingTensor {
        val xx = rewrittenForm(x.x)
        val y = rewrittenForm(x.y)
        val resultShape = x.a + x.b + x.d
        return if (xx !== x.x || y !== x.y)
            if (resultShape.isScalar)
                TracingScalar.Matmul(xx, y, x.a, x.b, x.c, x.d)
            else
                TracingTensor.Matmul(xx, y, x.a, x.b, x.c, x.d)
        else
            x
    }

    override fun visitOuterProduct(x: TracingTensor.OuterProduct): TracingTensor {
        val xx = rewrittenForm(x.x)
        val y = rewrittenForm(x.y)
        return if (xx !== x.x || y !== x.y)
            TracingTensor.OuterProduct(xx, y)
        else
            x
    }

    override fun visitSin(x: TracingTensor.Sin): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Sin(xx) else TracingTensor.Sin(xx)
        else
            x
    }

    override fun visitCos(x: TracingTensor.Cos): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Cos(xx) else TracingTensor.Cos(xx)
        else
            x
    }

    override fun visitTan(x: TracingTensor.Tan): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Tan(xx) else TracingTensor.Tan(xx)
        else
            x
    }

    override fun visitAtan(x: TracingTensor.Atan): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Atan(xx) else TracingTensor.Atan(xx)
        else
            x
    }

    override fun visitExp(x: TracingTensor.Exp): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Exp(xx) else TracingTensor.Exp(xx)
        else
            x
    }

    override fun visitLn(x: TracingTensor.Ln): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Ln(xx) else TracingTensor.Ln(xx)
        else
            x
    }

    override fun visitLgamma(x: TracingTensor.Lgamma): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Lgamma(xx) else TracingTensor.Lgamma(xx)
        else
            x
    }

    override fun visitDigamma(x: TracingTensor.Digamma): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Digamma(xx) else TracingTensor.Digamma(xx)
        else
            x
    }

    override fun visitPolygamma(x: TracingTensor.Polygamma): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Polygamma(x.n, xx) else TracingTensor.Polygamma(x.n, xx)
        else
            x
    }

    override fun visitSqrt(x: TracingTensor.Sqrt): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Sqrt(xx) else TracingTensor.Sqrt(xx)
        else
            x
    }

    override fun visitTanh(x: TracingTensor.Tanh): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Tanh(xx) else TracingTensor.Tanh(xx)
        else
            x
    }

    override fun visitMeld(x: TracingTensor.Meld): TracingTensor {
        val newValues = x.values.map { rewrittenForm(it) }
        val changed = x.values.zip(newValues).any { it.first !== it.second }
        return if (changed)
            TracingTensor.Meld(newValues)
        else
            x
    }

    override fun visitSplit(x: TracingTensor.Split): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Split(xx, x.shapes) else TracingTensor.Split(xx, x.shapes)
        else
            x
    }

    override fun visitSplitPart(x: TracingTensor.SplitPart): TracingTensor {
        val from = rewrittenForm(x.from)
        return if (from !== x.from)
            if (x.isScalar)
                TracingScalar.SplitPart(from, x.index)
            else
                TracingTensor.SplitPart(from, x.index, x.shape)
        else
            x
    }

    override fun visitConcat(x: TracingTensor.Concat): TracingTensor {
        val newSlices = x.slices.map { rewrittenForm(it) }
        val changed = x.slices.zip(newSlices).any { it.first !== it.second }
        return if (changed)
            TracingTensor.Concat(newSlices, x.axis)
        else
            x
    }

    override fun visitBroadcastTo(x: TracingTensor.BroadcastTo): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingTensor.BroadcastTo(xx, x.shape)
        else
            x
    }

    override fun visitConvImpl(x: TracingTensor.ConvImpl): TracingTensor {
        val signal = rewrittenForm(x.signal)
        val filter = rewrittenForm(x.filter)
        return if (signal !== x.signal || filter !== x.filter)
            TracingTensor.ConvImpl(signal, filter, x.hStride, x.vStride, x.padding)
        else
            x
    }

    override fun visitExpand(x: TracingTensor.Expand): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingTensor.Expand(xx, x.shape)
        else
            x
    }

    override fun visitFlip(x: TracingTensor.Flip): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingTensor.Flip(xx, x.axes)
        else
            x
    }

    override fun visitLogSoftmax(x: TracingTensor.LogSoftmax): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingTensor.LogSoftmax(xx, x.axis)
        else
            x
    }

    override fun visitLogSoftmaxGrad(x: TracingTensor.LogSoftmaxGrad): TracingTensor {
        val xx = rewrittenForm(x.x)
        val l = rewrittenForm(x.logSoftmax)
        val u = rewrittenForm(x.upstream)
        return if (xx !== x.x || l !== x.logSoftmax || u != x.upstream)
            TracingTensor.LogSoftmaxGrad(xx, x.axis, l, u)
        else
            x
    }

    override fun visitPow(x: TracingTensor.Pow): TracingTensor {
        val base = rewrittenForm(x.base)
        return if (base !== x.base)
            if (base is TracingScalar) TracingScalar.Pow(base, x.exponent) else TracingTensor.Pow(base, x.exponent)
        else
            x
    }

    override fun visitView1(x: TracingTensor.View1): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingTensor.View1(xx, x.indexes, x.shape)
        else
            x
    }

    override fun visitView2(x: TracingTensor.View2): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (x is DScalar)
                TracingScalar.View2(xx, x.index, x.axis)
            else
                TracingTensor.View2(xx, x.index, x.axis, x.shape)
        else
            x
    }

    override fun visitView3(x: TracingTensor.View3): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingTensor.View3(xx, x.index, x.axis)
        else
            x
    }

    override fun visitReshape(x: TracingTensor.Reshape): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingTensor.Reshape(xx, x.shape)
        else
            x
    }

    override fun visitReshapeToScalar(x: TracingScalar.ReshapeToScalar): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingScalar.ReshapeToScalar(xx)
        else
            x
    }

    override fun visitSqueeze(x: TracingTensor.Squeeze): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (x is DScalar) TracingScalar.Squeeze(xx) else TracingTensor.Squeeze(xx, x.axis)
        else
            x
    }

    override fun visitUnsqueeze(x: TracingTensor.Unsqueeze): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingTensor.Unsqueeze(xx, x.axis)
        else
            x
    }

    override fun visitTranspose(x: TracingTensor.Transpose): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingTensor.Transpose(xx, x.axes)
        else
            x
    }

    override fun visitRelu(x: TracingTensor.Relu): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Relu(xx) else TracingTensor.Relu(xx)
        else
            x
    }

    override fun visitReluGrad(x: TracingTensor.ReluGrad): TracingTensor {
        val xx = rewrittenForm(x.x)
        val ups = rewrittenForm(x.upstream)
        return if (xx !== x.x || ups != x.upstream)
            if (xx is TracingScalar) TracingScalar.ReluGrad(xx, ups) else TracingTensor.ReluGrad(xx, ups)
        else
            x
    }

    override fun visitSigmoid(x: TracingTensor.Sigmoid): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (xx is TracingScalar) TracingScalar.Sigmoid(xx) else TracingTensor.Sigmoid(xx)
        else
            x
    }

    override fun visitSum(x: TracingTensor.Sum): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            if (x is TracingScalar) TracingScalar.Sum(xx) else TracingTensor.Sum(xx, x.axes, x.keepDims)
        else
            x
    }

    override fun visitAvgPool(x: TracingTensor.AvgPool): TracingTensor {
        TODO("Not yet implemented")
    }

    override fun visitAvgPoolGrad(x: TracingTensor.AvgPoolGrad): TracingTensor {
        TODO("Not yet implemented")
    }

    override fun visitMaxPoolWithIndices(x: TracingTensor.MaxPoolWithIndices): TracingTensor {
        TODO("Not yet implemented")
    }

    override fun visitGather(x: TracingTensor.Gather): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingTensor.Gather(xx, x.indexes, x.axis, x.paddingIndex)
        else
            x
    }

    override fun visitGatherAtIndices(x: TracingTensor.GatherAtIndices): TracingTensor {
        TODO("Not yet implemented")
    }

    override fun visitScatter(x: TracingTensor.Scatter): TracingTensor {
        val xx = rewrittenForm(x.x)
        return if (xx !== x.x)
            TracingTensor.Scatter(xx, x.indexes, x.axis, x.shape, x.paddingIndex)
        else
            x
    }

    override fun visitScatterAtIndices(x: TracingTensor.ScatterAtIndices): TracingTensor {
        TODO("Not yet implemented")
    }

    override fun visitCompare(x: TracingTensor.Compare): TracingTensor {
        val left = rewrittenForm(x.left)
        val right = rewrittenForm(x.right)
        return if (x.left == left && x.right == right)
            x
        else if (x is TracingScalar)
            TracingScalar.Compare(left, right, x.comparison)
        else
            TracingTensor.Compare(left, right, x.comparison)
    }

    override fun visitIfThenElse(x: TracingTensor.IfThenElse): TracingTensor {
        val cond = rewrittenForm(x.cond)
        val whenTrue = rewrittenForm(x.whenTrue)
        val whenFalse = rewrittenForm(x.whenFalse)
        return if (x.cond == cond && x.whenTrue == whenTrue && x.whenFalse == whenFalse)
            x
        else if (x is TracingScalar)
            TracingScalar.IfThenElse(cond, whenTrue, whenFalse)
        else
            TracingTensor.IfThenElse(cond, whenTrue, whenFalse)
    }

    override fun visitRandomFloats(x: TracingTensor.RandomFloats): TracingTensor {
        val xx = rewrittenForm(x.key)
        return if (xx !== x.key)
            TracingTensor.RandomFloats(xx, x.shape)
        else
            x
    }

    override fun visitRandomSplit(x: TracingRandomKey.Split): Traceable {
        val key = rewrittenForm(x.key)
        return if (key != x.key) TracingRandomKey.Split(key, x.n) else x
    }

    override fun visitRandomSplitPart(x: TracingRandomKey.SplitPart): Traceable {
        val splitRewrite = rewrittenForm(x.split)
        return if (splitRewrite != x.split) TracingRandomKey.SplitPart(splitRewrite, x.splitIndex) else x
    }

    override fun visitRandomVariable(x: TracingRandomKey.Variable): Traceable {
        return x
    }
}
